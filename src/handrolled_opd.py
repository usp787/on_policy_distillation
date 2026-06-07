"""Appendix A — hand-rolled on-policy distillation, no TRL. See README Appendix A.

Mirrors the blog's pseudocode and is faster than TRL GKD because sampling uses
vLLM. Per step:

  1. SAMPLE  N student rollouts with vLLM (keep per-token logp_student).
  2. SCORE   feed the exact student token sequences to the teacher, read
             prompt_logprobs (one forward pass, no backprop) -> logp_teacher.
  3. ADVANTAGE  per-token reverse-KL: A_t = -(logp_student_t - logp_teacher_t).
             Discount 0 -> optimize each token independently (as the blog does).
  4. UPDATE  importance-sampling policy-gradient step on the student (+LoRA),
             via an HF forward/backward, weighted by A_t.

This is a teaching scaffold, not a tuned trainer. Swapping step 3's reverse-KL
advantage for the §9 verifiable reward turns this into a GRPO loop — which is the
whole "distillation is RL with a dense reward" point. Run behind sbatch only.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

SYSTEM_PROMPT = (
    "You are a careful mathematician. Solve the problem step by step, then give "
    "the final answer as a single integer inside \\boxed{}."
)


def load_prompts(path: str, tokenizer, limit: int | None):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit:
        rows = rows[:limit]
    prompts = []
    for ex in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["problem"]},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return prompts


def teacher_token_logprobs(teacher_llm, full_token_ids, prompt_len):
    """Run one teacher forward over the exact (prompt+completion) token ids and
    return per-completion-token logprobs via vLLM prompt_logprobs."""
    from vllm import SamplingParams

    # prompt_logprobs gives logprob of each token given its prefix when we pass
    # the full sequence as the "prompt" and generate 0 new tokens.
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    outs = teacher_llm.generate(prompt_token_ids=full_token_ids, sampling_params=sp)
    logps = []
    for out, plen in zip(outs, prompt_len):
        pl = out.prompt_logprobs  # list aligned to prompt tokens; [0] is None
        seq = []
        for tok_pos in range(plen, len(pl)):
            entry = pl[tok_pos]
            if entry is None:
                seq.append(0.0)
                continue
            # entry: {token_id: Logprob(...)}; take the realized token's logprob
            tok_id = full_token_ids[outs.index(out)][tok_pos]
            seq.append(entry[tok_id].logprob if tok_id in entry else 0.0)
        logps.append(seq)
    return logps


def main() -> None:
    p = argparse.ArgumentParser(description="Hand-rolled vLLM on-policy distillation.")
    p.add_argument("--config", default="configs/gkd.yaml")
    p.add_argument("--student", help="override student")
    p.add_argument("--teacher", help="override teacher")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--rollouts", type=int, default=8, help="N student rollouts/step")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--refresh-every", type=int, default=5,
                   help="reload student vLLM weights from the LoRA-merged student")
    p.add_argument("--prompt-limit", type=int, default=256)
    p.add_argument("--output-dir", default="checkpoints/handrolled_opd")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    student_id = args.student or cfg["student"]
    teacher_id = args.teacher or cfg["teacher"]

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(student_id, trust_remote_code=True)
    prompts = load_prompts(cfg["train_data"], tok, args.prompt_limit)

    # Trainable student (HF + LoRA) for the backward pass.
    student = AutoModelForCausalLM.from_pretrained(
        student_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    student = get_peft_model(student, LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]))
    student.train()
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)

    # Two vLLM engines: one samples from the student, one scores with the teacher.
    # (Two engines co-resident is the tight case; FP8 the teacher if VRAM is short.)
    student_engine = LLM(model=student_id, trust_remote_code=True,
                         gpu_memory_utilization=0.40, dtype="bfloat16")
    teacher_engine = LLM(model=teacher_id, trust_remote_code=True,
                         gpu_memory_utilization=0.40, dtype="bfloat16")

    sampling = SamplingParams(
        n=args.rollouts, temperature=1.0, top_p=0.95,
        max_tokens=args.max_new_tokens, logprobs=0,
    )

    import random

    for step in range(args.steps):
        prompt = random.choice(prompts)
        gen = student_engine.generate([prompt], sampling)[0]

        # Build full token sequences + capture student logprobs per completion token.
        prompt_ids = gen.prompt_token_ids
        full_ids, prompt_lens, student_logps = [], [], []
        for o in gen.outputs:
            comp_ids = list(o.token_ids)
            full_ids.append(list(prompt_ids) + comp_ids)
            prompt_lens.append(len(prompt_ids))
            # vLLM logprobs: list[dict] aligned to generated tokens
            lp = [next(iter(d.values())).logprob if d else 0.0 for d in (o.logprobs or [])]
            student_logps.append(lp)

        teacher_logps = teacher_token_logprobs(teacher_engine, full_ids, prompt_lens)

        # Per-token reverse-KL advantage + IS policy-gradient update.
        opt.zero_grad()
        total_loss = 0.0
        for fids, plen, s_lp, t_lp in zip(full_ids, prompt_lens, student_logps, teacher_logps):
            comp_ids = fids[plen:]
            n = min(len(comp_ids), len(s_lp), len(t_lp))
            if n == 0:
                continue
            ids = torch.tensor([fids], device="cuda")
            logits = student(ids).logits[0]  # (seq, vocab)
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            # logprob of realized completion token at each position
            cur_lp = torch.stack([
                logprobs[plen + i - 1, comp_ids[i]] for i in range(n)
            ])
            s_old = torch.tensor(s_lp[:n], device="cuda")
            adv = -(s_old - torch.tensor(t_lp[:n], device="cuda"))  # reverse-KL, per token
            ratio = torch.exp(cur_lp - s_old)                       # importance weight
            loss = -(ratio * adv.detach()).mean()
            loss.backward()
            total_loss += loss.item()
        opt.step()

        print(f"[opd] step {step:>3}  loss {total_loss / max(1, args.rollouts):+.4f}")

        if (step + 1) % args.refresh_every == 0:
            # Refresh the sampler's weights from the LoRA-merged student.
            merged = student.merge_and_unload()
            tmp = Path(args.output_dir) / "_merged_tmp"
            merged.save_pretrained(tmp)
            tok.save_pretrained(tmp)
            # Cheapest portable refresh: rebuild the student engine from the merge.
            del student_engine
            student_engine = LLM(model=str(tmp), trust_remote_code=True,
                                 gpu_memory_utilization=0.40, dtype="bfloat16")
            # re-wrap LoRA on a fresh copy so training continues
            student = get_peft_model(
                AutoModelForCausalLM.from_pretrained(
                    str(tmp), torch_dtype=torch.bfloat16, trust_remote_code=True
                ).cuda(),
                LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]),
            )
            student.train()
            opt = torch.optim.AdamW(
                [p for p in student.parameters() if p.requires_grad], lr=args.lr
            )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[opd] saved -> {out}")


if __name__ == "__main__":
    main()
