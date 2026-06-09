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
# Yes / No / Not-Given examiner. The rules below target the exact mistakes the 4B
# judges kept making: reading "not enough information" as "No", converse/inverse
# fallacies, and confusing existential ("some") with universal ("all") claims.
SYSTEM_YNN = """\
You are a strict formal logic examiner. Use ONLY the given premises, definitions, and the target statement. Do not use outside knowledge.
Your task: Decide whether the premises logically entail the target statement.
Answer rules:
* ANSWER: Yes = the premises prove the statement.
* ANSWER: No = the premises prove the negation of the statement.
* ANSWER: Not Given = the premises do not prove the statement and do not prove its negation.
* Very important: Do NOT answer "No" just because the statement is not proven. Lack of proof = "Not Given", not "No".
Reasoning rules:
1. Use direct implication:
   * From "A -> B" and "A", infer "B".
2. Use contraposition:
   * From "A -> B", infer "not B -> not A".
   * From "not A -> not B", infer "B -> A".
   * Example: "If not paid, not allowed" allows "allowed -> paid".
3. Do NOT use the converse:
   * From "A -> B", do NOT infer "B -> A".
   * Example: "If certified, trained" does not mean "if trained, certified".
4. Do NOT use the inverse:
   * From "A -> B", do NOT infer "not A -> not B".
5. Quantifier rules:
   * "All X are Y" means every X is Y.
   * "Everyone is Y" applies to every person, including any subgroup such as students/employees.
   * "Some X are Y" or "at least one X is Y" does NOT mean all X are Y.
   * "Some X are Y" does NOT contradict "All X are Y".
   * If the question asks "all", one example is not enough.
   * If the question asks "some", one proven example is enough.
6. Negation rules:
   * To answer "No", you must prove the opposite.
   * For "All X are Y", the opposite is "Some X are not Y".
   * For "Some X are Y", the opposite is "No X are Y".
   * If the opposite is not proven, answer "Not Given".
7. Necessary/sufficient condition rules:
   * "A if B" means "B -> A".
   * "A only if B" means "A -> B".
   * "Only A are B" means "B -> A".
   * "A requires B" means "A -> B".
   * "Cannot B unless A" means "B -> A".
   * "Cannot B if not A" means "not A -> not B", so by contraposition "B -> A".
8. Wording and definitions:
   * Treat different wordings as different predicates unless the Definitions section says they are equivalent.
   * If a Definitions section exists, treat each definition as a logical equivalence.
   * Do not invent new definitions or aliases.
   * For this benchmark, if a premise says "must", "required to", "have to", or "need to", treat it as a factual rule unless the problem explicitly says it is only an obligation.
9. For statements containing "because":
   * Verify that the final claim is entailed.
   * Also verify that the stated reason supports the claim through the premises.
   * If the claim is true but the stated reason is not supported, answer "Not Given".
Before answering, silently check:
* Is the statement directly proven?
* Is the statement proven by contraposition?
* Is only the converse being used? If yes, reject it.
* Is the model confusing "some" with "all"?
* Is "No" being used when the correct answer is only "Not Given"?
Reply in EXACTLY this format: ANSWER: <Yes | No | Not Given> WHY: <one short sentence, at most 30 words, citing premise numbers>"""

# Multiple-choice examiner — same logic discipline, picks a single option letter.
SYSTEM_MCQ = """\
You are a strict formal logic examiner. Use ONLY the given premises, definitions, and answer options. Do not use outside knowledge.
Your task: Pick the single best option that is logically entailed by the premises.
Answer rules:
* Pick an option only if it must be true from the premises.
* If no option follows, answer "Unknown".
* Do not pick an option just because it sounds plausible.
* Do not pick an option that uses the converse of a premise.
* Do not pick an option that changes "some" into "all" or "all" into "some" incorrectly.
* Do not pick an option that adds an unsupported existence claim.
* Do not pick a contradiction unless the premises themselves entail that contradiction.
* If multiple options seem true, choose the most direct non-tautological option supported by the premises.
Reasoning rules:
1. Direct implication is valid:
   * From "A -> B" and "A", infer "B".
2. Contraposition is valid:
   * From "A -> B", infer "not B -> not A".
   * From "not A -> not B", infer "B -> A".
3. Converse is invalid:
   * From "A -> B", do NOT infer "B -> A".
4. Inverse is invalid:
   * From "A -> B", do NOT infer "not A -> not B".
5. Quantifiers:
   * "Some" does not mean "all".
   * "At least one" does not mean "all".
   * "Everyone" applies to all relevant individuals and subgroups.
   * If an option says "all", the premises must prove all.
   * If an option says "some", the premises must prove at least one.
6. Necessary/sufficient condition rules:
   * "A if B" means "B -> A".
   * "A only if B" means "A -> B".
   * "Only A are B" means "B -> A".
   * "A requires B" means "A -> B".
   * "Cannot B unless A" means "B -> A".
   * "Cannot B if not A" means "not A -> not B", so "B -> A".
7. Wording and definitions:
   * Treat different wordings as different predicates unless the Definitions section says they are equivalent.
   * If a Definitions section exists, treat definitions as logical equivalences.
   * Do not invent aliases.
   * For this benchmark, "must", "required to", "have to", and "need to" are treated as factual rules unless explicitly stated otherwise.
Before answering, silently test each option:
* Is it directly entailed?
* Is it entailed by contraposition?
* Does it require an invalid converse?
* Does it change "some" to "all"?
* Does it add unsupported facts?
Reply in EXACTLY this format: ANSWER: <letter | Unknown> WHY: <one short sentence, at most 30 words, citing premise numbers>"""


def system_for(record: Record) -> str:
    return SYSTEM_MCQ if record.answer_type == AnswerType.MCQ else SYSTEM_YNN


def definitions_for(record: Record) -> str:
    """The dataset folds everything into premises-NL, so there is no separate
    definitions block — surface 'None'. (A `raw['definitions']` list, if a future
    dataset ever carries one, is rendered here instead.)"""
    defs = record.raw.get("definitions") if isinstance(record.raw, dict) else None
    if defs:
        if isinstance(defs, (list, tuple)):
            return "\n" + "\n".join(f"- {d}" for d in defs)
        return f" {defs}"
    return " None"


def build_user(record: Record) -> str:
    """The user turn, matching the system prompts' template:

        Premises:  (numbered from 1, so the model can cite "premise 7")
        Definitions: None
        Question:  <fixed task for YNN | the stem for MCQ>
        Statement: <the claim to test>        (YNN only)
        Options:   A. … B. …                  (MCQ only)
    """
    prem = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(record.premises_nl))
    parts = [f"Premises:\n{prem}", f"Definitions:{definitions_for(record)}"]
    if record.answer_type == AnswerType.MCQ and record.options:
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(record.options))
        parts += [f"Question: {record.question_nl}", f"Options:\n{opts}"]
    else:
        # Yes/No/Not-Given: the dataset's question string IS the target statement;
        # the "Question" slot carries the fixed decision task.
        parts += [
            "Question: Does the statement logically follow from the premises? "
            "Answer Yes, No, or Not Given.",
            f"Statement: {record.question_nl}",
        ]
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
_WHY_INLINE = re.compile(r"\bWHY\b\s*[:\-=]", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Thinking-mode / chain-of-thought stripping. Models tag reasoning differently —
# Qwen uses <think>…</think>, Gemma can use <start_of_thought>…<end_of_thought>,
# others <thinking>…</thinking>; some emit only the closing tag. LFM2.5 reasons
# in prose with no tags. Whatever the model does, the real guard against reasoning
# leaking into the WHY is parse_reply only reading text AFTER the ANSWER line.
_THINK_BLOCK = re.compile(
    r"<think>.*?</think>|<thinking>.*?</thinking>|<start_of_thought>.*?<end_of_thought>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_CLOSE = re.compile(r"^.*?(?:</think>|</thinking>|<end_of_thought>)", re.IGNORECASE | re.DOTALL)
_THINK_TAGS = re.compile(r"</?think>|</?thinking>|</?start_of_thought>|</?end_of_thought>", re.IGNORECASE)


def strip_thinking(raw: str) -> str:
    """Remove a closed reasoning block so it can't leak into the parsed answer.
    The verbatim `raw` is kept for the logs; only the parsing copy is cleaned."""
    raw = _THINK_BLOCK.sub(" ", raw or "")
    low = raw.lower()
    if "</think>" in low or "</thinking>" in low or "<end_of_thought>" in low:
        raw = _THINK_CLOSE.sub(" ", raw)
    return _THINK_TAGS.sub(" ", raw)


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
    clean = strip_thinking(raw or "")
    m = _ANSWER_LINE.search(clean)
    ans_token, inline_why = None, ""
    if m:
        ans_line = m.group(1).strip().splitlines()[0].strip()
        # "ANSWER: Yes WHY: …" on one line — split the answer from the inline WHY.
        parts = _WHY_INLINE.split(ans_line, maxsplit=1)
        ans_token = parts[0].strip()
        if len(parts) > 1:
            inline_why = parts[1].strip()

    canon, display = canonicalize(ans_token, record) if ans_token else (None, "")
    if canon is None:
        canon, display = _scavenge_answer(clean, record)

    # WHY: prefer the inline WHY on the ANSWER line, then a WHY: line AFTER the
    # answer, then the prose AFTER the answer. NEVER the text before the answer —
    # in thinking mode that prose is the model's reasoning, not its explanation.
    after = clean[m.end():] if m else clean
    if inline_why:
        why = compact_why(inline_why)
    else:
        mw = _WHY_LINE.search(after)
        why = compact_why(mw.group(1) if mw else after)
    return canon, (display or (canon or "")), why
