"""Accuracy read-out (optional, only with --show-gold).

Predictions are already canonical (MCQ → letter; YNN → Yes/No/Unknown), as is
gold (see data_load._canon_gold), so a direct equality is the right comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schema import AnswerType, FinalAnswer, Record


def is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    return pred.strip().lower() == gold.strip().lower()


@dataclass
class TypeStats:
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class Report:
    overall: TypeStats = field(default_factory=TypeStats)
    by_type: dict[str, TypeStats] = field(default_factory=dict)
    agreed: int = 0
    escalated: int = 0

    def add(self, atype: AnswerType, correct: bool, agreed: bool) -> None:
        self.overall.total += 1
        self.overall.correct += int(correct)
        slot = self.by_type.setdefault(atype.value, TypeStats())
        slot.total += 1
        slot.correct += int(correct)
        self.agreed += int(agreed)
        self.escalated += int(not agreed)

    def to_dict(self) -> dict:
        return {
            "overall": {"total": self.overall.total, "correct": self.overall.correct,
                        "accuracy": self.overall.accuracy},
            "by_type": {k: {"total": v.total, "correct": v.correct, "accuracy": v.accuracy}
                        for k, v in self.by_type.items()},
            "agreed": self.agreed,
            "escalated_to_8b": self.escalated,
        }


def score(records: list[Record], finals: dict[str, FinalAnswer]) -> Report:
    rep = Report()
    for r in records:
        f = finals.get(r.id)
        if f is None:
            rep.add(r.answer_type, correct=False, agreed=False)
            continue
        rep.add(r.answer_type, is_correct(f.answer, r.answer), f.agreed)
    return rep
