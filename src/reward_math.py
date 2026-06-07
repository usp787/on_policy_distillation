"""Verifiable reward for RLVR (Phase 1, GRPO). See README §9.

1.0 if the model's final boxed answer matches ground truth, else 0.0 — sparse
and verifiable. math-verify handles the symbolic/numeric equivalence so e.g.
"1/2" == "0.5" == "\\frac{1}{2}".
"""
from math_verify import parse, verify


def math_reward(completions, answer, **kwargs):
    """TRL GRPO reward signature: list[completion], list[ground-truth] -> list[float].

    `completions` may be plain strings or chat-style lists of message dicts
    (depending on whether the dataset is conversational); normalize both.
    """
    out = []
    for c, gt in zip(completions, answer):
        text = _as_text(c)
        try:
            out.append(1.0 if verify(parse(gt), parse(text)) else 0.0)
        except Exception:  # noqa: BLE001 — a malformed parse is just a wrong answer
            out.append(0.0)
    return out


def _as_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # [{"role": ..., "content": ...}, ...]
        return "".join(
            m.get("content", "") for m in completion if isinstance(m, dict)
        )
    return str(completion)
