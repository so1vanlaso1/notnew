"""The cascade decision logic.

Per record:
  1. Both 4B judges answer (Qwen3.5-4B and Gemma-E2B).
  2. If their canonical answers AGREE → that is the final answer, and Qwen's
     own explanation is used.
  3. If they DISAGREE → the record is deferred; later the 8B model (Gemma-E4B)
     decides and explains.

The VRAM invariant (≤ two 4B models OR one 8B model) is enforced by the caller
(`run_cascade.py`), which loads/unloads in phases. This module is pure logic.
"""

from __future__ import annotations

from schema import FinalAnswer, ModelReply, Record
from prompts import build_user, parse_reply, system_for

DECIDER_AGREE = "agreement(4B Qwen+Gemma-E2B)"
DECIDER_BIG = "fallback(8B Gemma-E4B)"
DECIDER_BIG_UNAVAILABLE = "fallback unavailable -> Qwen-4B"
DECIDER_NONE = "no answer"

CONF_AGREE = 0.90
CONF_BIG = 0.70
CONF_QWEN_ONLY = 0.40
CONF_NONE = 0.0


def query(model, record: Record, max_new_tokens: int = 256) -> ModelReply:
    """Ask one model for its answer + compact explanation on one record.

    Never raises: a generation failure becomes a ModelReply with answer=None and
    the error captured in `raw` (so it still shows up in the model-I/O log, and
    the record simply escalates instead of aborting the whole batch)."""
    system = system_for(record)
    user = build_user(record)
    # Render the prompt up front so it is logged even if generation throws.
    try:
        prompt = (model.render(system, user) if hasattr(model, "render")
                  else f"[SYSTEM]\n{system}\n\n[USER]\n{user}")
    except Exception:  # noqa: BLE001
        prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
    raw, elapsed = "", 0.0
    try:
        raw, prompt2, elapsed = model.generate(system, user, max_new_tokens=max_new_tokens)
        prompt = prompt2 or prompt
    except Exception as e:  # noqa: BLE001 — keep the batch alive, log the failure
        raw = f"<generation error: {type(e).__name__}: {e}>"
    canon, display, why = parse_reply(raw, record)
    return ModelReply(
        model_label=model.label,
        model_id=model.model_id,
        prompt=prompt,
        raw=raw,
        answer=canon,
        answer_display=display or (canon or ""),
        explanation=why,
        elapsed_s=elapsed,
    )


def judges_agree(a: ModelReply, b: ModelReply) -> bool:
    """Two judges agree only when both produced the SAME, non-empty canonical
    answer. If either failed to produce a parseable answer, they do not agree
    (so the record escalates to the 8B model)."""
    return a.answer is not None and a.answer == b.answer


def finalize_agreed(record: Record, qwen: ModelReply, gemma: ModelReply) -> FinalAnswer:
    """Both 4B judges agreed → keep the answer; explanation comes from Qwen."""
    return FinalAnswer(
        id=record.id,
        answer_type=record.answer_type,
        answer=qwen.answer,
        answer_display=qwen.answer_display,
        explanation=qwen.explanation,
        decider=DECIDER_AGREE,
        agreed=True,
        confidence=CONF_AGREE,
        replies=[qwen, gemma],
        elapsed_s=qwen.elapsed_s + gemma.elapsed_s,
    )


def finalize_with_big(record: Record, qwen: ModelReply, gemma: ModelReply,
                      big: ModelReply) -> FinalAnswer:
    """Judges disagreed → the 8B model decides AND explains."""
    if big.answer is not None:
        answer, display, expl, decider, conf = (
            big.answer, big.answer_display, big.explanation, DECIDER_BIG, CONF_BIG
        )
    else:
        # 8B produced nothing parseable — fall back to Qwen's stored answer.
        answer, display, expl, decider, conf = (
            qwen.answer, qwen.answer_display, qwen.explanation,
            DECIDER_BIG_UNAVAILABLE, CONF_QWEN_ONLY,
        )
    return FinalAnswer(
        id=record.id,
        answer_type=record.answer_type,
        answer=answer,
        answer_display=display or (answer or ""),
        explanation=expl,
        decider=decider,
        agreed=False,
        confidence=conf if answer is not None else CONF_NONE,
        replies=[qwen, gemma, big],
        elapsed_s=qwen.elapsed_s + gemma.elapsed_s + big.elapsed_s,
    )


def finalize_big_unloadable(
    record: Record, qwen: ModelReply, gemma: ModelReply,
    big_model_id: str = "", load_error: str | None = None,
) -> FinalAnswer:
    """The 8B model could not be loaded at all — keep Qwen's stored answer so the
    record still gets a (lower-confidence) prediction instead of being dropped.

    A synthetic 8B reply recording the load failure is appended so the attempt is
    still visible in the model-I/O log ("log everything")."""
    replies = [qwen, gemma]
    if load_error is not None:
        replies.append(ModelReply(
            model_label="Gemma-E4B(8B)",
            model_id=big_model_id,
            prompt="<8B escalation attempted — the two 4B judges disagreed>",
            raw=f"<8B model could not be loaded: {load_error}>",
            answer=None, answer_display="", explanation="", elapsed_s=0.0,
        ))
    return FinalAnswer(
        id=record.id,
        answer_type=record.answer_type,
        answer=qwen.answer,
        answer_display=qwen.answer_display,
        explanation=qwen.explanation,
        decider=DECIDER_BIG_UNAVAILABLE,
        agreed=False,
        confidence=CONF_QWEN_ONLY if qwen.answer is not None else CONF_NONE,
        replies=replies,
        elapsed_s=qwen.elapsed_s + gemma.elapsed_s,
    )
