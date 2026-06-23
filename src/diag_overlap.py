"""OPD overlap-ratio diagnostic. See README §10 and arXiv:2604.13016 (THUNLP, "Rethinking
On-Policy Distillation: Phenomenology, Mechanism, and Recipe").

Question it answers: does our flat Phase-2 result match the paper's *failed-run signature*?
The paper shows OPD fails when either (i) student/teacher thinking patterns mismatch (low
top-k overlap), or (ii) the teacher carries no NEW knowledge (high overlap, but its per-token
distribution is already ~matched by the student, so there is nothing to transfer). Our 4B and
30B are same-family / same-mode / same 2507 release, so we predict HIGH overlap but a
near-zero overlap-advantage and small entropy gap — i.e. condition (ii), not (i).

What it measures, averaged over STUDENT-visited states (the student's own sampled tokens):
  - overlap_ratio       |topk_s ∩ topk_t| / k                       paper Eq.6  (mismatch≈0.28; healthy 0.55–0.91)
  - overlap_advantage   Σ_{v∈O} p̃_s(v)(log p̃_t(v) − log p̃_s(v))     paper Eq.7  (= −KL on the shared set; ≤0, →0 = aligned)
  - entropy_gap         |H_teacher − H_student|                     paper Eq.8  (→0 = matched confidence)
  - overlap mass        student/teacher prob mass on the shared top-k  (paper: the top-k carries ~0.97–0.99 of the mass)

Mechanism: we sample rollouts from the student, then run ONE teacher-forcing forward pass of
each model over the student's own token ids (== evaluating both on student-visited prefixes,
paper §2.2) and read the next-token distributions. No training, no backprop — read-only.

Both models MUST share the Qwen3 151,936-token vocab (so the same token id means the same
token in both top-k sets); asserted at load. Writes results/diag_overlap_<student>__vs__<teacher>.json.
Run via sbatch (slurm/diag_overlap.sbatch) — needs a GPU. Light enough for 1× H100, <1 h.
"""
import argparse
import json
import os
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Verbatim from eval_aime.py / train_gkd.py — the student must roll out in the SAME format it
# was distilled/evaluated in, so the visited states match the OPD training distribution.
SYSTEM_PROMPT = (
    "You are a careful mathematician. Solve the problem step by step, then give "
    "the final answer as a single integer inside \\boxed{}."
)


def safe_model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip("/"))


def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3: non-thinking, matches all phases
    )


def load_problems(path: str, n: int, seed: int) -> list[str]:
    """Load {"problem", ...} rows (same schema as data/train_math.json / aime*.json)."""
    from datasets import load_dataset

    ds = load_dataset("json", data_files=path, split="train")
    if "problem" not in ds.column_names:
        raise KeyError(f"{path} has no 'problem' column (cols={ds.column_names}).")
    if len(ds) > n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return [ex["problem"] for ex in ds]


def main() -> None:
    p = argparse.ArgumentParser(description="OPD overlap-ratio diagnostic (arXiv:2604.13016).")
    p.add_argument("--student", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="fresh student = the INITIAL overlap (the paper's predictor). Point at "
                        "$HF_HOME/opd_merged/phase2_distill_merged to probe the post-distill student.")
    p.add_argument("--teacher", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                   help="same teacher as configs/gkd.yaml (FP8 → ~31 GB, fits beside the 4B on one H100).")
    p.add_argument("--data", default="data/train_math.json",
                   help="prompt source; default = the OPD training distribution (README §7).")
    p.add_argument("--num-prompts", type=int, default=64,
                   help="rollouts to sample; even 64×~800 tok ≈ 50k positions is plenty for stable means.")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="rollout cap. 1024 keeps the job <1 h on an H100; OPD trained at 2048 — raise "
                        "to 2048 to match exactly if you move to an H200.")
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--topk", type=int, default=16, help="k for the top-k sets (paper default k=16).")
    p.add_argument("--temperature", type=float, default=0.7, help="paper/eval sampling temp.")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-positions-per-seq", type=int, default=0,
                   help="0 = use all completion positions; >0 caps per-rollout positions to bound compute.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/diag_overlap")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    eps = 1e-12
    k = args.topk

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad so generated tokens start at a fixed column per batch

    # Student in bf16; teacher with dtype="auto" so transformers honors the FP8 quantization_config
    # (~31 GB resident vs ~61 GB bf16). device_map="cuda" places each fully on the single GPU.
    print(f"[diag] loading student {args.student} (bf16)")
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype="bfloat16", trust_remote_code=True, device_map="cuda"
    ).eval()
    print(f"[diag] loading teacher {args.teacher} (auto/FP8)")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype="auto", trust_remote_code=True, device_map="cuda"
    ).eval()

    # Same-vocab is what makes the per-token top-k overlap well-defined (README §2).
    assert student.config.vocab_size == teacher.config.vocab_size, (
        f"vocab mismatch: student {student.config.vocab_size} vs teacher "
        f"{teacher.config.vocab_size} — top-k overlap is only meaningful with a shared vocab."
    )

    problems = load_problems(args.data, args.num_prompts, args.seed)
    prompts = [build_prompt(pr, tok) for pr in problems]
    print(f"[diag] {len(prompts)} prompts from {args.data}")

    # ---- 1) sample student rollouts (on-policy states) ------------------------------------
    records = []  # (full_ids[L], prompt_len) per rollout
    for i in range(0, len(prompts), args.gen_batch_size):
        batch = prompts[i : i + args.gen_batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")  # template already has the control tokens
        with torch.inference_mode():
            gen = student.generate(
                **enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
            )
        width = enc["input_ids"].shape[1]
        for j in range(len(batch)):
            real_prompt = enc["input_ids"][j][enc["attention_mask"][j].bool()]  # strip left pad
            comp = gen[j][width:]                                               # generated tokens
            eos = (comp == tok.eos_token_id).nonzero()
            if eos.numel():
                comp = comp[: eos[0, 0]]            # drop EOS and any trailing pad
            if comp.numel() == 0:
                continue
            full = torch.cat([real_prompt, comp])
            records.append((full.cpu(), int(real_prompt.shape[0])))
        print(f"[diag] generated {len(records)}/{len(prompts)} rollouts")

    # ---- 2) score: per-token top-k overlap / advantage / entropy on student-visited states -
    agg = {key: 0.0 for key in
           ("overlap_ratio", "entropy_s", "entropy_t", "entropy_gap",
            "mass_s_topk", "mass_t_topk", "mass_s_overlap", "mass_t_overlap")}
    n_pos = 0          # positions counted for the above
    adv_sum, adv_pos = 0.0, 0   # advantage averaged only over positions with ≥1 overlap token
    n_trunc = 0

    for idx, (full_cpu, prompt_len) in enumerate(records):
        if prompt_len < 1:
            continue
        full = full_cpu.to("cuda").unsqueeze(0)
        with torch.inference_mode():
            ls = student(full).logits[0]   # [L, V]
            lt = teacher(full).logits[0]   # [L, V]
        # logits at positions [prompt_len-1 : L-1] predict the completion tokens [prompt_len : L]
        sl = slice(prompt_len - 1, full.shape[1] - 1)
        ls, lt = ls[sl].float(), lt[sl].float()
        if args.max_positions_per_seq and ls.shape[0] > args.max_positions_per_seq:
            n_trunc += 1
            ls, lt = ls[: args.max_positions_per_seq], lt[: args.max_positions_per_seq]
        T = ls.shape[0]
        if T == 0:
            continue

        lp_s, lp_t = ls.log_softmax(-1), lt.log_softmax(-1)
        p_s, p_t = lp_s.exp(), lp_t.exp()

        # entropy (nats) and gap
        H_s = -(p_s * lp_s).sum(-1)            # [T]
        H_t = -(p_t * lp_t).sum(-1)
        # top-k sets and their intersection
        topk_s = ls.topk(k, dim=-1).indices    # [T,k]
        topk_t = lt.topk(k, dim=-1).indices
        in_t = (topk_s.unsqueeze(2) == topk_t.unsqueeze(1)).any(2)  # [T,k] which student-topk ids are also teacher-topk
        overlap_ratio = in_t.sum(-1).float() / k                    # [T]

        # probability mass on each model's own top-k, and on the shared (overlap) set O ⊆ topk_s
        ps_topk = p_s.gather(-1, topk_s)       # [T,k] student prob at student top-k ids
        pt_topk = p_t.gather(-1, topk_s)        # [T,k] teacher prob at the SAME ids
        mass_s_topk = ps_topk.sum(-1)
        mass_t_topk = p_t.gather(-1, topk_t).sum(-1)
        mass_s_overlap = (ps_topk * in_t).sum(-1)
        mass_t_overlap = (pt_topk * in_t).sum(-1)

        # overlap-token advantage = Σ_{v∈O} p̃_s(v)(log p̃_t(v) − log p̃_s(v)) = −KL(p̃_s‖p̃_t) on O (≤0)
        ps_O = ps_topk * in_t
        pt_O = pt_topk * in_t
        ps_n = ps_O / ps_O.sum(-1, keepdim=True).clamp_min(eps)
        pt_n = pt_O / pt_O.sum(-1, keepdim=True).clamp_min(eps)
        term = ps_n * (pt_n.clamp_min(eps).log() - ps_n.clamp_min(eps).log())
        adv = (term * in_t).sum(-1)            # [T]
        valid = in_t.any(-1)                   # positions with a non-empty overlap set

        agg["overlap_ratio"] += overlap_ratio.sum().item()
        agg["entropy_s"] += H_s.sum().item()
        agg["entropy_t"] += H_t.sum().item()
        agg["entropy_gap"] += (H_t - H_s).abs().sum().item()
        agg["mass_s_topk"] += mass_s_topk.sum().item()
        agg["mass_t_topk"] += mass_t_topk.sum().item()
        agg["mass_s_overlap"] += mass_s_overlap.sum().item()
        agg["mass_t_overlap"] += mass_t_overlap.sum().item()
        n_pos += T
        adv_sum += adv[valid].sum().item()
        adv_pos += int(valid.sum().item())

        del ls, lt, lp_s, lp_t, p_s, p_t
        if (idx + 1) % 8 == 0:
            print(f"[diag] scored {idx + 1}/{len(records)} rollouts ({n_pos} positions)")

    if n_pos == 0:
        raise SystemExit("[diag] no student-visited positions collected — check generation/data.")

    m = {key: round(val / n_pos, 5) for key, val in agg.items()}
    m["overlap_advantage"] = round(adv_sum / max(1, adv_pos), 6)
    overlap = m["overlap_ratio"]

    # ---- 3) interpretation (soft; lead with overlap_ratio, the metric with clear paper ranges) ----
    verdict = []
    if overlap >= 0.6:
        verdict.append(
            f"overlap_ratio={overlap:.3f} is HIGH (paper: mismatch≈0.28, healthy 0.55–0.91) → "
            "thinking patterns are compatible; condition (i) is NOT the blocker, and an SFT cold-start "
            "(the paper's fix for (i)) would likely NOT help us.")
    else:
        verdict.append(
            f"overlap_ratio={overlap:.3f} is LOW → thinking-pattern mismatch (paper condition i); "
            "an off-policy SFT cold-start on teacher rollouts would be the lever to try.")
    verdict.append(
        f"entropy_gap={m['entropy_gap']:.3f} nats (→0 = matched confidence); "
        f"overlap_advantage={m['overlap_advantage']:.5f} (≤0, →0 = student already matches the teacher "
        "on shared tokens).")
    verdict.append(
        "READ: high overlap + near-zero advantage + small entropy gap ⇒ the teacher's per-token signal "
        "is already ~satisfied by the student ⇒ little NEW knowledge to transfer (paper condition ii) ⇒ "
        "consistent with our flat Phase-2 (50.00→48.75). Low/mismatched values would instead indict (i).")

    results = {
        "student": args.student, "teacher": args.teacher, "data": args.data,
        "num_rollouts": len(records), "num_positions": n_pos, "topk": k,
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
        "metrics": m,
        "paper_reference": {
            "source": "arXiv:2604.13016 (THUNLP, Rethinking On-Policy Distillation)",
            "overlap_ratio": "mismatch/failed ≈0.28; healthy runs 0.55 → rise to ~0.91",
            "overlap_advantage": "≤0; converges toward 0 in successful runs, stagnant in failed runs",
            "entropy_gap": "narrows toward 0 in successful runs",
            "topk_mass": "top-k typically carries 0.97–0.99 of the probability mass",
        },
        "interpretation": verdict,
    }
    out_path = Path(f"{args.out}_{safe_model_tag(args.student)}__vs__{safe_model_tag(args.teacher)}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[diag] ===== overlap diagnostic =====")
    for key in ("overlap_ratio", "overlap_advantage", "entropy_gap", "entropy_s", "entropy_t",
                "mass_s_topk", "mass_t_topk", "mass_s_overlap", "mass_t_overlap"):
        val = m.get(key, results["metrics"].get(key))
        print(f"[diag]   {key:18s} = {val}")
    print(f"[diag] positions={n_pos} rollouts={len(records)} (adv over {adv_pos} pos)")
    for line in verdict:
        print(f"[diag] → {line}")
    print(f"[diag] wrote {out_path}")


if __name__ == "__main__":
    main()
