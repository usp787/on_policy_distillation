"""Phase 2/3 — on-policy distillation via trl GKDTrainer. See README §10, §11.

lmbda=1.0 (100% student-generated, on-policy) + beta=1.0 (reverse KL) ≈ the
blog's setup. Teacher = Phase-1 GRPO checkpoint (self-distill) or an external
Qwen3-8B/14B (Phase 3). Student is always a FRESH Qwen3-4B-Instruct-2507.

NOTE: GKD is experimental in TRL — import path / arg names move between versions
(README Appendix B). We try a couple of known locations.
"""
import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

try:  # newer TRL
    from trl.experimental.gkd import GKDConfig, GKDTrainer
except ImportError:  # older TRL exposed it at the top level
    from trl import GKDConfig, GKDTrainer

SYSTEM_PROMPT = (
    "You are a careful mathematician. Solve the problem step by step, then give "
    "the final answer as a single integer inside \\boxed{}."
)


def build_dataset(path: str):
    """GKD trains on prompts (it samples completions on-policy from the student).
    Provide chat-formatted `messages`; the trainer applies the chat template."""
    ds = load_dataset("json", data_files=path, split="train")

    def to_messages(ex):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["problem"]},
            ]
        }

    return ds.map(to_messages, remove_columns=ds.column_names)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2/3 on-policy distillation (GKD).")
    p.add_argument("--config", default="configs/gkd.yaml")
    p.add_argument("--student", help="override config student")
    p.add_argument("--teacher", help="override config teacher (Phase 3 swap)")
    p.add_argument("--output-dir", help="override config output_dir")
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    student_id = args.student or cfg["student"]
    teacher_id = args.teacher or cfg["teacher"]

    tok = AutoTokenizer.from_pretrained(student_id, trust_remote_code=True)
    ds = build_dataset(cfg["train_data"])

    lora = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])

    student = AutoModelForCausalLM.from_pretrained(
        student_id, dtype="bfloat16", trust_remote_code=True
    )

    # This TRL version's GKDTrainer takes the teacher as an explicit `teacher_model` arg
    # (it does NOT auto-load from GKDConfig.teacher_model_name_or_path). Load it as a bf16
    # object — the 30B-A3B teacher in fp32 would be ~120 GB and OOM. No device_map: the
    # trainer's accelerator.prepare_model() places it on the GPU as a frozen eval model.
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, dtype="bfloat16", trust_remote_code=True
    )

    gkd_kwargs = dict(cfg["gkd"])
    if args.output_dir:
        gkd_kwargs["output_dir"] = args.output_dir
    if args.max_steps is not None:
        gkd_kwargs["max_steps"] = args.max_steps
    gkd_cfg = GKDConfig(**gkd_kwargs)

    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=gkd_cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
    )
    # Auto-resume: TRL GKD on HF-generate is generate-bound and a 600-step run can't
    # finish one 8h slot (job 7497131 died at step 59/600). save_steps=25 leaves a
    # resumable checkpoint; if output_dir already holds a checkpoint-N, continue from it.
    # Just resubmit the sbatch to chain across slots. None -> fresh start (no checkpoint yet).
    resume = None
    if Path(gkd_cfg.output_dir).is_dir():
        resume = get_last_checkpoint(gkd_cfg.output_dir)
    if resume:
        print(f"[gkd] resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(gkd_cfg.output_dir)
    print(f"[gkd] saved adapter -> {gkd_cfg.output_dir}  (teacher={teacher_id})")


if __name__ == "__main__":
    main()
