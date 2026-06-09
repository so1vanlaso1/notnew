"""The cascade decision logic — weighted soft voting across stages.

Every selected model answers every record (the 8B model is no longer fired only
on disagreement — it votes on every question). Each model contributes a weight to
the label it picked; the label with the most total weight wins:

    4B judge  → weight 1.0      (Qwen3.5-4B, Gemma-E2B)
    8B model  → weight 1.5      (Gemma-E4B, LFM2.5-8B-A1B)

So a single 8B (1.5) outvotes one disagreeing 4B (1.0), but two agreeing 4B
judges (2.0) outvote a lone 8B (1.5). Confidence is the winning vote's share of
the total weight, and the explanation is taken from the strongest model that
voted for the winning label (an 8B's reasoning wins over a 4B's).

The VRAM invariant (≤ two 4B models OR one 8B model resident) is enforced by the
caller (`run_cascade.py`), which loads/runs/unloads one stage at a time. This
module is pure logic.
"""

from __future__ import annotations

from schema import FinalAnswer, ModelReply, Record
from prompts import build_user, parse_reply, system_for

# Default vote weights by model size class.
WEIGHT_4B = 1.0
WEIGHT_8B = 1.5

DECIDER_UNANIMOUS = "unanimous vote"
DECIDER_VOTE = "weighted vote"
DECIDER_NONE = "no parseable answer"

_EPS = 1e-9


def query(model, record: Record, max_new_tokens: int = 256) -> ModelReply:
    """Ask one model for its answer + compact explanation on one record.

    Never raises: a generation failure becomes a ModelReply with answer=None and
    the error captured in `raw` (so it still shows up in the model-I/O log, and
    the record simply gets one fewer vote instead of aborting the whole batch)."""
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
        weight=float(getattr(model, "vote_weight", WEIGHT_4B)),
        model_class=getattr(model, "model_class", "4b"),
    )


# ── Weighted soft vote ────────────────────────────────────────────────────────
def tally(replies: list[ModelReply]) -> dict[str, float]:
    """Sum each model's weight onto the label it voted for (None votes abstain)."""
    scores: dict[str, float] = {}
    for r in replies:
        if r.answer is None:
            continue
        scores[r.answer] = scores.get(r.answer, 0.0) + r.weight
    return scores


def _break_tie(winners: list[str], replies: list[ModelReply]) -> str:
    """Tied total weight → defer to the single strongest individual vote; if
    still tied, to the latest stage that voted for it (more authoritative). For a
    pure 2×4B split (1.0 vs 1.0) this resolves to the later judge."""
    best_label, best_key = winners[0], None
    for idx, r in enumerate(replies):
        if r.answer in winners:
            key = (r.weight, idx)  # higher weight first, then later in the run
            if best_key is None or key > best_key:
                best_key, best_label = key, r.answer
    return best_label


def _pick_explainer(label: str, replies: list[ModelReply]) -> ModelReply | None:
    """The explanation for the final answer comes from the strongest model that
    voted for `label` and actually wrote a WHY; falls back to the strongest such
    model even if its WHY is empty."""
    voters = [r for r in replies if r.answer == label]
    if not voters:
        return None
    # Highest weight first, then later stage (an 8B's reasoning beats a 4B's).
    voters_ranked = sorted(
        range(len(voters)), key=lambda i: (voters[i].weight, i), reverse=True
    )
    for i in voters_ranked:
        if voters[i].explanation.strip():
            return voters[i]
    return voters[voters_ranked[0]]


def finalize_by_vote(record: Record, replies: list[ModelReply]) -> FinalAnswer:
    """Combine every model's vote on one record into the final answer."""
    scores = tally(replies)
    elapsed = sum(r.elapsed_s for r in replies)
    if not scores:
        # Nobody produced a parseable answer.
        return FinalAnswer(
            id=record.id, answer_type=record.answer_type, answer=None,
            answer_display="", explanation="", decider=DECIDER_NONE,
            agreed=False, confidence=0.0, replies=replies, scores=scores,
            elapsed_s=elapsed,
        )

    total = sum(scores.values())
    best = max(scores.values())
    winners = [lbl for lbl, w in scores.items() if abs(w - best) < _EPS]
    win = winners[0] if len(winners) == 1 else _break_tie(winners, replies)

    voters = [r for r in replies if r.answer is not None]
    unanimous = len(scores) == 1 and len(voters) == len(replies)
    explainer = _pick_explainer(win, replies)

    return FinalAnswer(
        id=record.id,
        answer_type=record.answer_type,
        answer=win,
        answer_display=(explainer.answer_display if explainer else win),
        explanation=(explainer.explanation if explainer else ""),
        decider=DECIDER_UNANIMOUS if unanimous else DECIDER_VOTE,
        agreed=unanimous,
        confidence=best / total if total else 0.0,
        replies=replies,
        scores=scores,
        elapsed_s=elapsed,
    )
