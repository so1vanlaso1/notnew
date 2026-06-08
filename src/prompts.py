"""Prompt construction + answer parsing/normalization.

The model is asked to reply in a strict, answer-first format:

    ANSWER: <A|B|C|… | Yes | No | Not Given>
    WHY: <one short sentence>

so the answer is trivially separable from the (compact) explanation and the two
judges can be compared on a canonical token.
"""

from __future__ import annotations

import re

from schema import AnswerType, Record

LETTERS = "ABCDEFGH"

# ── System prompts ──────────────────────────────────────────────────────────
SYSTEM_MCQ = (
    "You are a careful logic examiner. Answer the multiple-choice question using "
    "ONLY the given premises — no outside knowledge. Pick the single best option.\n"
    "Reply in EXACTLY this format, answer first, nothing before it:\n"
    "ANSWER: <letter>\n"
    "WHY: <one short sentence, at most 30 words, citing premise numbers>\n"
    "If no option follows from the premises, write 'ANSWER: Unknown'."
)

SYSTEM_YNN = (
    "You are a careful logic examiner. Answer the question using ONLY the given "
    "premises — no outside knowledge.\n"
    "Decide: 'Yes' if the premises entail the statement, 'No' if they contradict "
    "it, 'Not Given' if the premises neither entail nor contradict it.\n"
    "Reply in EXACTLY this format, answer first, nothing before it:\n"
    "ANSWER: <Yes | No | Not Given>\n"
    "WHY: <one short sentence, at most 30 words, citing premise numbers>"
)


def system_for(record: Record) -> str:
    return SYSTEM_MCQ if record.answer_type == AnswerType.MCQ else SYSTEM_YNN


def build_user(record: Record) -> str:
    """The user turn: numbered premises (from 1, to match the dataset's
    'Premises 7 and 8…' style), the question, and MCQ options when present."""
    prem = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(record.premises_nl))
    parts = [f"Premises:\n{prem}", "", f"Question: {record.question_nl}"]
    if record.answer_type == AnswerType.MCQ and record.options:
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(record.options))
        parts += ["", f"Options:\n{opts}"]
    return "\n".join(parts)


# ── Answer normalization ─────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower()).strip()


# Order matters: "Unknown" markers are checked before the bare yes/no startswith,
# so "Not Given" / "no information" don't get misread as "No".
_UNKNOWN_MARKERS = (
    "not given", "notgiven", "not-given", "unknown", "uncertain", "undetermined",
    "cannot be determined", "cannot determine", "can't be determined",
    "insufficient", "no information", "no info", "neither", "none of",
)


def canonicalize_ynn(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower().rstrip(".")
    for m in _UNKNOWN_MARKERS:
        if m in v:
            return "Unknown"
    if v.startswith("yes") or v in ("true", "correct", "entailed"):
        return "Yes"
    if v.startswith("no") or v in ("false", "incorrect", "contradicted"):
        return "No"
    return None


_LEADING_LETTER = re.compile(r"^[\(\[]?([A-Ha-h])[\)\.\:\]]?(?:\s|$)")


def canonicalize_mcq(value: str | None, options: list[str] | None) -> tuple[str | None, str | None]:
    """Return (letter, display). letter is 'A'..'H' or 'Unknown' or None."""
    if not value:
        return None, None
    v = value.strip()
    low = v.lower().rstrip(".")
    if low in ("unknown", "none", "none of the above", "not given", "n/a", "na"):
        return "Unknown", "Unknown"
    m = _LEADING_LETTER.match(v)
    if m:
        letter = m.group(1).upper()
        idx = ord(letter) - 65
        disp = f"{letter}. {options[idx]}" if options and idx < len(options) else letter
        return letter, disp
    # Fall back to matching the option text itself. Exact (normalized) match
    # first; then GUARDED containment only — a short token like "no" must not
    # match an option merely because it appears inside a word ("ho-no-rs",
    # "insufficie-n-t"). Require a substantial, similar-length overlap.
    if options:
        nv = _norm(v)
        if nv:
            for i, o in enumerate(options):
                if _norm(o) == nv:
                    return LETTERS[i], f"{LETTERS[i]}. {o}"
            for i, o in enumerate(options):
                no = _norm(o)
                if not no:
                    continue
                shorter, longer = sorted((len(nv), len(no)))
                if (nv in no or no in nv) and shorter >= 6 and shorter >= 0.6 * longer:
                    return LETTERS[i], f"{LETTERS[i]}. {o}"
    return None, None


def canonicalize(value: str | None, record: Record) -> tuple[str | None, str]:
    """Canonicalize a raw answer token for a record. Returns (canon, display)."""
    if record.answer_type == AnswerType.MCQ:
        letter, disp = canonicalize_mcq(value, record.options)
        return letter, (disp or (value or "").strip())
    canon = canonicalize_ynn(value)
    display = {"Yes": "Yes", "No": "No", "Unknown": "Not Given"}.get(canon or "", (value or "").strip())
    return canon, display


# ── Reply parsing ─────────────────────────────────────────────────────────────
_ANSWER_LINE = re.compile(r"ANSWER\s*[:\-=]\s*(.+)", re.IGNORECASE)
_WHY_LINE = re.compile(r"WHY\s*[:\-=]\s*(.+)", re.IGNORECASE | re.DOTALL)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def compact_why(text: str, max_chars: int = 240) -> str:
    """First one–two sentences, whitespace-collapsed, capped — keep it short."""
    text = " ".join((text or "").split())
    if not text:
        return ""
    sents = _SENT_SPLIT.split(text)
    out = sents[0]
    if len(out) < 80 and len(sents) > 1:
        out = (out + " " + sents[1]).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


def _scavenge_answer(raw: str, record: Record) -> tuple[str | None, str]:
    """No clean ANSWER line — best-effort recovery from the body."""
    if record.answer_type == AnswerType.MCQ:
        m = re.search(r"\boption\s+([A-Ha-h])\b", raw)
        if m:
            return canonicalize_mcq(m.group(1), record.options)
        m = re.search(r"\b([A-H])\b", raw)  # a lone capital letter
        if m:
            return canonicalize_mcq(m.group(1), record.options)
        return None, ""
    return canonicalize(raw, record)


def parse_reply(raw: str, record: Record) -> tuple[str | None, str, str]:
    """Return (canonical_answer, display, compact_explanation)."""
    raw = raw or ""
    ans_token = None
    m = _ANSWER_LINE.search(raw)
    if m:
        ans_token = m.group(1).strip().splitlines()[0].strip()

    canon, display = canonicalize(ans_token, record) if ans_token else (None, "")
    if canon is None:
        canon, display = _scavenge_answer(raw, record)

    why = ""
    mw = _WHY_LINE.search(raw)
    if mw:
        why = compact_why(mw.group(1))
    if not why:
        # Strip any ANSWER line and use the remaining prose as the explanation.
        body = _ANSWER_LINE.sub("", raw)
        why = compact_why(body)
    return canon, (display or (canon or "")), why
