"""Phase 1 — RLVR via trl.GRPOTrainer (vLLM-accelerated). See README §9.

Same loop shape as distillation, but the reward is the sparse verifiable
math_reward (1.0 if the boxed answer is right, else 0.0). This is where the
clean, attributable AIME gain comes from.
"""
import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward_math import math_reward

SYSTEM_PROMPT = (
    "You are a careful mathematician. Solve the problem step by step, then give "
    "the final answer as a single integer inside \\boxed{}."
)


def build_dataset(path: str, tokenizer):
    """Load {"problem","answer"} rows into the conversational format GRPO expects:
    a `prompt` (list of chat messages) + the `answer` column the reward reads."""
    ds = load_dataset("json", data_files=path, split="train")

    def to_prompt(ex):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["problem"]},
            ],
            "answer": str(ex["answer"]),
        }

    return ds.map(to_prompt, remove_columns=[c for c in ds.column_names if c != "answer"])


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1 RLVR (GRPO).")
    p.add_argument("--config", default="configs/grpo.yaml")
    p.add_argument("--model", help="override config model")
    p.add_argument("--max-steps", type=int, help="override config max_steps")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_id = args.model or cfg["model"]

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    ds = build_dataset(cfg["train_data"], tok)

    lora = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])

    grpo_kwargs = dict(cfg["grpo"])
    if args.max_steps is not None:
        grpo_kwargs["max_steps"] = args.max_steps
    grpo_cfg = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=math_reward,
        args=grpo_cfg,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(grpo_cfg.output_dir)
    print(f"[grpo] saved adapter -> {grpo_cfg.output_dir}")


if __name__ == "__main__":
    main()
