"""Phase 2/3 — on-policy distillation via trl GKDTrainer. See README §10, §11.

lmbda=1.0 (100% student-generated, on-policy) + beta=1.0 (reverse KL) ≈ the
blog's setup. Teacher = Phase-1 GRPO checkpoint (self-distill) or an external
Qwen3-8B/14B (Phase 3). Student is always a FRESH Qwen3-4B-Instruct-2507.

NOTE: GKD is experimental in TRL — import path / arg names move between versions
(README Appendix B). We try a couple of known locations.
"""
import argparse
import json
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
    # (it does NOT auto-load from GKDConfig.teacher_model_name_or_path). No device_map: the
    # trainer's accelerator.prepare_model() places it on the GPU as a frozen eval model.
    # The default teacher is the FP8 (block-128) 30B checkpoint — load with dtype="auto" so
    # transformers honors its fp8 quantization_config (~31 GB resident vs ~61 GB bf16, which
    # OOM'd the dense KL). Forcing bf16 here would dequantize and defeat the point. A non-FP8
    # teacher (e.g. a Phase-3 swap) has no quantization_config, so "auto" gives it bf16 anyway.
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, dtype="auto", trust_remote_code=True
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
    # Guarded auto-resume. save_steps leaves a checkpoint so a timed-out job can continue
    # (resubmit the sbatch). BUT a checkpoint from a DIFFERENT config silently restores its
    # optimizer/scheduler — that footgun once resumed a stale 1e-6-LR checkpoint into a 5e-6
    # run and trained at the wrong LR. So only resume when the run signature matches; if a
    # checkpoint exists under a different/absent signature, refuse and make the user decide.
    out = Path(gkd_cfg.output_dir)
    sig_keys = ["learning_rate", "max_steps", "per_device_train_batch_size",
                "gradient_accumulation_steps", "max_new_tokens"]
    signature = json.dumps(
        {"student": student_id, "teacher": teacher_id,
         **{k: gkd_kwargs.get(k) for k in sig_keys}},
        sort_keys=True,
    )
    sig_path = out / "opd_run_signature.json"
    last_ckpt = get_last_checkpoint(str(out)) if out.is_dir() else None
    resume = None
    if last_ckpt:
        prev = sig_path.read_text(encoding="utf-8") if sig_path.exists() else None
        if prev == signature:
            resume = last_ckpt
            print(f"[gkd] resuming from {resume} (config signature matches)")
        else:
            raise SystemExit(
                f"[gkd] REFUSING to resume: {last_ckpt} exists but its run config differs "
                f"from the current one (or predates signature tracking) — resuming would "
                f"inherit the wrong optimizer/scheduler (e.g. a stale LR). Either "
                f"`rm -rf {out}` to start fresh, or restore the matching config.\n"
                f"  current : {signature}\n  on disk : {prev}"
            )
    out.mkdir(parents=True, exist_ok=True)
    sig_path.write_text(signature, encoding="utf-8")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(gkd_cfg.output_dir)
    print(f"[gkd] saved adapter -> {gkd_cfg.output_dir}  (teacher={teacher_id})")


if __name__ == "__main__":
    main()
