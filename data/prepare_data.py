"""Download + cache the math training prompts and the AIME eval sets.

Run via sbatch on the cluster (it hits the network / HF hub) — never locally.
See README §7. Produces JSON files under data/ with a uniform schema:

    {"problem": <str>, "answer": <ground-truth str>}

Training prompts are downsampled to ~3-5k (distillation reuses prompts happily).
AIME'24 + AIME'25 are kept strictly held out for eval.
"""
import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent


def _boxed_answer(text: str) -> str | None:
    """Pull the final \\boxed{...} content out of a reference solution, if present."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
        j += 1
    return None


def _write(rows: list[dict], path: Path) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {len(rows):>5} rows -> {path}")


# --------------------------------------------------------------------------- #
# Training prompts (verifiable-answer math)
# --------------------------------------------------------------------------- #
def prepare_train(name: str, n: int, seed: int) -> None:
    """Build the training set. `name` selects the source dataset.

    Each source has a slightly different schema, so we normalize here. If your
    chosen dataset isn't covered, add a branch — the only contract downstream is
    {"problem", "answer"}.
    """
    print(f"[train] loading {name} (target {n} prompts)")

    if name == "HuggingFaceH4/MATH":
        ds = load_dataset(name, split="train")
        rows = []
        for ex in ds:
            ans = _boxed_answer(ex.get("solution", ""))
            if ans:
                rows.append({"problem": ex["problem"], "answer": ans})

    elif name == "agentica-org/DeepScaleR-Preview-Dataset":
        ds = load_dataset(name, split="train")
        rows = [
            {"problem": ex["problem"], "answer": str(ex["answer"]).strip()}
            for ex in ds
            if ex.get("answer") is not None
        ]

    elif name == "open-r1/OpenR1-Math-220k":
        ds = load_dataset(name, "default", split="train")
        rows = [
            {"problem": ex["problem"], "answer": str(ex["answer"]).strip()}
            for ex in ds
            if ex.get("answer")
        ]

    else:
        raise ValueError(f"Unknown train dataset: {name}")

    # Downsample to keep one job small (README §7).
    ds_rows = load_dataset_from_rows(rows).shuffle(seed=seed)
    rows = ds_rows.select(range(min(n, len(ds_rows))))
    rows = [dict(r) for r in rows]
    _write(rows, DATA_DIR / "train_math.json")


def load_dataset_from_rows(rows):
    from datasets import Dataset

    return Dataset.from_list(rows)


# --------------------------------------------------------------------------- #
# Eval sets (held out): AIME'24 + AIME'25
# --------------------------------------------------------------------------- #
def _normalize_aime(ex: dict) -> dict | None:
    # Field names differ across mirrors; probe the common ones.
    problem = ex.get("problem") or ex.get("question") or ex.get("Problem")
    answer = ex.get("answer") or ex.get("Answer") or ex.get("solution")
    if problem is None or answer is None:
        return None
    return {"problem": str(problem).strip(), "answer": str(answer).strip()}


def prepare_eval() -> None:
    sources = {
        "aime24": ("HuggingFaceH4/aime_2024", None, "train"),
        "aime25": ("yentinglin/aime_2025", None, "train"),
    }
    for tag, (repo, config, split) in sources.items():
        print(f"[eval] loading {tag} <- {repo}")
        try:
            ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        except Exception as e:  # noqa: BLE001
            print(f"  !! failed to load {repo} ({e}). Try an alternative mirror "
                  f"(e.g. opencompass/AIME2025) and re-run.")
            continue
        rows = [r for r in (_normalize_aime(ex) for ex in ds) if r]
        _write(rows, DATA_DIR / f"{tag}.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare math train + AIME eval data.")
    p.add_argument("--train-dataset", default="HuggingFaceH4/MATH",
                   help="source for training prompts")
    p.add_argument("--n-train", type=int, default=4000,
                   help="number of training prompts to keep (~3-5k recommended)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    if not args.skip_train:
        prepare_train(args.train_dataset, args.n_train, args.seed)
    if not args.skip_eval:
        prepare_eval()
    print("done.")


if __name__ == "__main__":
    main()
