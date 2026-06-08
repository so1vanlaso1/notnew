"""Run logging — timestamped .txt files under Result/, in the spirit of the old
pipeline's run_*.txt / *_model_io.txt.

Three artifacts share one timestamp:
  run_cascade_<stamp>.txt            human-readable per-record summary + accuracy
  run_cascade_<stamp>_model_io.txt   the FULL prompt + raw output of EVERY model
                                     invocation (both 4B judges, and the 8B model
                                     whenever it was fired)
  run_cascade_<stamp>.json           machine-readable predictions
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from schema import AnswerType, FinalAnswer, Record


def result_dir(root: Path) -> Path:
    d = root / "Result"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wrap(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _kind(rec: Record) -> str:
    return "MCQ" if rec.answer_type == AnswerType.MCQ else "YNN"


def write_run_summary(
    path: Path, header: dict, records: list[Record], finals: dict[str, FinalAnswer],
    n_correct: int, n_scored: int, elapsed_s: float,
) -> Path:
    lines: list[str] = ["=" * 78, f"Run summary — {datetime.now():%Y-%m-%d %H:%M:%S}"]
    for k, v in header.items():
        lines.append(f"{k}={v}")
    acc = f"{n_correct}/{n_scored} = {n_correct / n_scored:.1%}" if n_scored else "n/a (no --show-gold)"
    lines.append(f"records={len(records)}   accuracy={acc}   elapsed={elapsed_s:.1f}s")
    lines.append("=" * 78)

    for i, rec in enumerate(records, 1):
        f = finals.get(rec.id)
        lines.append("")
        lines.append(f"### [{i}/{len(records)}] {rec.id}   [{_kind(rec)}]")
        lines.append(f"Q: {_wrap(rec.question_nl)}")
        if rec.answer_type == AnswerType.MCQ and rec.options:
            for j, o in enumerate(rec.options):
                lines.append(f"   {chr(ord('A') + j)}. {_wrap(o, 90)}")
        if f is None:
            lines.append("  (no verdict)")
            continue
        # Both judges' raw votes.
        for rep in f.replies[:2]:
            lines.append(f"  judge {rep.model_label:<16} -> {rep.answer!r}  ({_wrap(rep.answer_display, 70)})")
        lines.append(f"  agreement: {'YES' if f.agreed else 'NO — escalated to 8B'}")
        if len(f.replies) >= 3:
            big = f.replies[2]
            lines.append(f"  8B {big.model_label:<19} -> {big.answer!r}  ({_wrap(big.answer_display, 70)})")
        lines.append(f"  >> ANSWER: {f.answer_display!r}   (confidence {f.confidence:.2f}, via {f.decider})")
        if f.explanation:
            lines.append(f"     WHY: {_wrap(f.explanation, 200)}")
        if f.gold is not None:
            ok = (f.answer or "").strip().lower() == f.gold.strip().lower()
            lines.append(f"     gold: {f.gold!r}   [{'CORRECT' if ok else 'WRONG'}]")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_model_io(path: Path, records: list[Record], finals: dict[str, FinalAnswer]) -> Path:
    """Verbatim record of every model call: the exact prompt sent and the exact
    raw text returned, in invocation order (judge A, judge B, then 8B if fired)."""
    lines = [f"MODEL I/O — {datetime.now():%Y-%m-%d %H:%M:%S}",
             "Every model invocation, verbatim (prompt in, raw text out).",
             "=" * 78]
    for i, rec in enumerate(records, 1):
        f = finals.get(rec.id)
        lines.append("")
        lines.append(f"### [{i}] {rec.id}   [{_kind(rec)}]")
        if f is None:
            lines.append("  (no model output)")
            continue
        for n, rep in enumerate(f.replies, 1):
            lines.append("")
            lines.append(f"-- MODEL {n}: {rep.model_label}  ({rep.model_id})  "
                         f"[{rep.elapsed_s:.2f}s] --")
            lines.append("  IN  (prompt) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            lines.append(_indent(rep.prompt))
            lines.append("  OUT (raw)    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            lines.append(_indent(rep.raw))
            lines.append(f"  PARSED: answer={rep.answer!r}  display={rep.answer_display!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + ln for ln in (text or "").splitlines()) or (prefix + "(empty)")


def predictions_dict(records: list[Record], finals: dict[str, FinalAnswer]) -> dict:
    out: dict[str, dict] = {}
    for rec in records:
        f = finals.get(rec.id)
        if f is None:
            continue
        out[rec.id] = {
            "answer_type": rec.answer_type.value,
            "answer": f.answer,
            "answer_display": f.answer_display,
            "explanation": f.explanation,
            "agreed": f.agreed,
            "decider": f.decider,
            "confidence": f.confidence,
            "gold": f.gold,
            "elapsed_s": round(f.elapsed_s, 3),
            "votes": [
                {"model": r.model_label, "answer": r.answer, "display": r.answer_display}
                for r in f.replies
            ],
        }
    return out


def write_predictions_json(path: Path, records: list[Record], finals: dict[str, FinalAnswer]) -> Path:
    path.write_text(
        json.dumps(predictions_dict(records, finals), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
