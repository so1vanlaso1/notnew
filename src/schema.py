"""Data structures shared across the cascade pipeline.

Deliberately uses plain dataclasses (no pydantic) so the data/prompt/logging
layers import with zero heavy dependencies — the model layer is the only part
that needs torch/transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnswerType(str, Enum):
    # The dataset's non-MCQ label set is {Yes, No, Unknown}. "Unknown" is what
    # the prompt surfaces to the user as "Not Given".
    YES_NO_UNKNOWN = "yes_no_unknown"
    MCQ = "mcq"
    OPEN_ENDED = "open_ended"


@dataclass
class Record:
    """One question to answer. `answer`/`options` come from the dataset; only
    `premises_nl` + `question_nl` (+ parsed MCQ `options`) are ever shown to a
    model — the gold `answer` is held back for optional scoring."""

    id: str
    premises_nl: list[str]
    question_nl: str
    answer_type: AnswerType
    answer: str | None = None          # gold, canonicalized; NEVER fed to a model
    options: list[str] | None = None   # MCQ option texts, in A,B,C… order
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelReply:
    """Exactly what one model did on one record — kept verbatim for the logs."""

    model_label: str       # e.g. "Qwen-4B" / "Gemma-E2B" / "Gemma-E4B(8B)"
    model_id: str          # HF repo id (or "stub:…")
    prompt: str            # the exact text fed to the model (rendered template)
    raw: str               # the exact raw completion text
    answer: str | None     # canonical: "A"/"B"/… (MCQ) or "Yes"/"No"/"Unknown"
    answer_display: str     # human readable, e.g. "A. Sophia qualifies…"
    explanation: str
    elapsed_s: float = 0.0


@dataclass
class FinalAnswer:
    id: str
    answer_type: AnswerType
    answer: str | None         # canonical final answer
    answer_display: str
    explanation: str
    decider: str               # which path produced it (see cascade.DECIDER_*)
    agreed: bool               # did the two 4B judges agree?
    confidence: float
    replies: list[ModelReply] = field(default_factory=list)
    gold: str | None = None    # only set when --show-gold
    elapsed_s: float = 0.0
