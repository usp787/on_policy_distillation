"""Phase-0 baseline + re-eval harness. See README §8.

vLLM batch-generate (non-thinking, boxed-answer prompt), score with math-verify,
write results/<out>_<model>.json. This is the *one ruler* used across all phases,
so the prompt/parser/length must stay fixed — don't trust numbers from other
harnesses (README §1).

Supports avg@k (sample k completions per problem, average the 0/1 score) to damp
the variance of a 30-problem test (README §12).
"""
import argparse
import json
import os
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Non-thinking, boxed-answer instruction. Kept verbatim across phases.
SYSTEM_PROMPT = (
    "You are a careful mathematician. Solve the problem step by step, then give "
    "the final answer as a single integer inside \\boxed{}."
)


def load_bench(tag: str) -> list[dict]:
    path = DATA_DIR / f"{tag}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run data/prepare_data.py (via sbatch) first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompts(problems: list[str], tokenizer) -> list[str]:
    prompts = []
    for prob in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prob},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3: force non-thinking (README §13/Appx B)
            )
        )
    return prompts


def score(completions: list[str], gold: str) -> float:
    """Mean 0/1 exact-match over k samples for one problem (avg@k)."""
    from math_verify import parse, verify

    hits = 0
    for c in completions:
        try:
            if verify(parse(gold), parse(c)):
                hits += 1
        except Exception:  # noqa: BLE001
            pass
    return hits / max(1, len(completions))


def safe_model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip("/"))


def resolve_model_path(model: str) -> str:
    """vLLM can't load a bare PEFT/LoRA adapter directory (no config.json / base
    weights). If `model` is such a dir, merge the adapter into its base model and
    return the merged dir; otherwise return `model` unchanged.

    The merge runs on CPU so it doesn't compete with vLLM for GPU memory. The
    merged copy is cached next to the adapter (<dir>_merged) and reused on re-runs.
    All phase checkpoints (1/2/3) are LoRA adapters, so this path is the norm for
    re-eval, not the exception.
    """
    p = Path(model)
    adapter_cfg = p / "adapter_config.json"
    is_adapter = p.is_dir() and adapter_cfg.exists() and not (p / "config.json").exists()
    if not is_adapter:
        return model  # plain HF id or a full-weight local dir

    merged_dir = p.parent / f"{p.name}_merged"
    if (merged_dir / "config.json").exists():
        print(f"[eval] reusing merged model at {merged_dir}")
        return str(merged_dir)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = json.loads(adapter_cfg.read_text(encoding="utf-8"))["base_model_name_or_path"]
    print(f"[eval] merging LoRA adapter {p} into base {base} -> {merged_dir} (CPU)")
    base_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base_model, str(p)).merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    AutoTokenizer.from_pretrained(str(p), trust_remote_code=True).save_pretrained(str(merged_dir))
    del merged, base_model
    print(f"[eval] merged model written -> {merged_dir}")
    return str(merged_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM AIME eval with math-verify scoring.")
    p.add_argument("--model", required=True, help="HF id or local checkpoint path")
    p.add_argument("--bench", default="aime24,aime25",
                   help="comma-separated bench tags (must match data/<tag>.json)")
    p.add_argument("--max-new-tokens", type=int, default=8192,
                   help="8192 is the fixed ruler: at 2048 the 4B truncated ~37%+ of "
                        "AIME solutions (16.7%% vs 43.3%% on aime25). Keep this constant "
                        "across all phases so deltas are comparable.")
    p.add_argument("--k", type=int, default=1, help="samples per problem (avg@k)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy (use >0 with --k>1 for avg@k)")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--out", default="results/phase0",
                   help="output prefix; file becomes <out>_<model>.json")
    p.add_argument("--dtype", default="auto", help="vLLM dtype, e.g. auto/bfloat16/fp8")
    p.add_argument("--gpu-mem-frac", type=float, default=0.90)
    p.add_argument("--debug-samples", type=int, default=0,
                   help="if >0, dump the first N (problem, gold, completion) per bench "
                        "to <out>_<model>.debug.json for manual inspection")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = resolve_model_path(args.model)  # merges a LoRA adapter dir if needed
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_frac,
    )

    temperature = args.temperature
    if args.k > 1 and temperature == 0.0:
        temperature = 0.7  # greedy + k>1 is pointless; nudge to sampling
        print(f"[eval] k={args.k} with temperature 0 -> bumping to {temperature}")

    sampling = SamplingParams(
        n=args.k,
        temperature=temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    results = {"model": args.model, "max_new_tokens": args.max_new_tokens,
               "k": args.k, "temperature": temperature, "benchmarks": {}}

    all_correct, all_total = 0.0, 0
    debug_dump = []
    for tag in [b.strip() for b in args.bench.split(",") if b.strip()]:
        bench = load_bench(tag)
        problems = [ex["problem"] for ex in bench]
        golds = [ex["answer"] for ex in bench]
        prompts = build_prompts(problems, tok)

        outputs = llm.generate(prompts, sampling)
        per_problem = []
        n_trunc, n_comp = 0, 0  # how many completions hit the length cap (no EOS)
        for ex, gold, out in zip(bench, golds, outputs):
            comps = [o.text for o in out.outputs]
            n_trunc += sum(1 for o in out.outputs if o.finish_reason == "length")
            n_comp += len(out.outputs)
            s = score(comps, gold)
            per_problem.append({"answer": gold, "score": s})
            if len(debug_dump) < args.debug_samples:
                o0 = out.outputs[0]
                debug_dump.append({
                    "bench": tag, "problem": ex["problem"], "gold": gold,
                    "score": s, "finish_reason": o0.finish_reason,
                    "completion": o0.text,
                })

        acc = sum(pp["score"] for pp in per_problem) / max(1, len(per_problem))
        trunc_pct = round(100 * n_trunc / max(1, n_comp), 1)
        results["benchmarks"][tag] = {
            "n": len(per_problem),
            "pass@1" if args.k == 1 else f"avg@{args.k}": round(100 * acc, 2),
            "truncated_pct": trunc_pct,  # high => max_new_tokens too small (answers cut off)
            "per_problem": per_problem,
        }
        all_correct += sum(pp["score"] for pp in per_problem)
        all_total += len(per_problem)
        print(f"[eval] {tag}: {100 * acc:.2f}%  (n={len(per_problem)}, "
              f"truncated={trunc_pct}%)")

    if all_total:
        combined = 100 * all_correct / all_total
        results["combined"] = {"n": all_total, "pass@1": round(combined, 2)}
        print(f"[eval] combined: {combined:.2f}%  (n={all_total})")

    out_path = Path(f"{args.out}_{safe_model_tag(args.model)}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_path}")

    if debug_dump:
        dbg_path = out_path.with_suffix(".debug.json")
        dbg_path.write_text(json.dumps(debug_dump, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] wrote {len(debug_dump)} sample completions -> {dbg_path}")


if __name__ == "__main__":
    main()
