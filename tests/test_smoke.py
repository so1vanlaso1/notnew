"""No-GPU smoke tests: data loading, prompt/answer normalization, and the full
cascade wiring via StubModel (both the agreement and the 8B-escalation branch).

Run:  python -m pytest NEWpipeline/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cascade import (  # noqa: E402
    finalize_agreed, finalize_big_unloadable, finalize_with_big, judges_agree, query,
)
from chat_model import StubModel  # noqa: E402
from data_load import load_records, parse_mcq_question  # noqa: E402
from prompts import canonicalize_mcq, canonicalize_ynn, parse_reply  # noqa: E402
from schema import AnswerType, Record  # noqa: E402

DATA = ROOT / "Logic_Based_Educational_Queries.json"


def test_dataset_loads_and_classifies():
    recs = load_records(DATA)
    assert len(recs) > 500
    kinds = {r.answer_type for r in recs}
    assert AnswerType.MCQ in kinds and AnswerType.YES_NO_UNKNOWN in kinds
    # gold is canonical: MCQ → single letter, YNN → Yes/No/Unknown
    for r in recs:
        if r.answer is None:
            continue
        if r.answer_type == AnswerType.MCQ:
            # a single option letter, or "Unknown" (none-of-the-above)
            assert r.answer == "Unknown" or (len(r.answer) == 1 and r.answer.isalpha())
        else:
            assert r.answer in {"Yes", "No", "Unknown"}


def test_mcq_question_split():
    stem, opts = parse_mcq_question("Pick one\nA. first\nB. second\nC. third")
    assert stem == "Pick one"
    assert opts == ["first", "second", "third"]


def test_ynn_normalization():
    assert canonicalize_ynn("Yes") == "Yes"
    assert canonicalize_ynn("no.") == "No"
    assert canonicalize_ynn("Not Given") == "Unknown"
    assert canonicalize_ynn("unknown") == "Unknown"
    assert canonicalize_ynn("no information in the premises") == "Unknown"
    assert canonicalize_ynn("banana") is None


def test_mcq_normalization():
    opts = ["alpha", "beta", "gamma"]
    assert canonicalize_mcq("B", opts)[0] == "B"
    assert canonicalize_mcq("(C) gamma", opts)[0] == "C"
    assert canonicalize_mcq("beta", opts)[0] == "B"
    assert canonicalize_mcq("Unknown", opts)[0] == "Unknown"


def test_mcq_no_substring_false_match():
    # A model that wrongly emits "No" on an MCQ must NOT match an option merely
    # because "no" appears inside a word (e.g. "honors").
    opts = ["Sophia is eligible for the program",
            "Sophia needs to pass to get an honors diploma",
            "John's GPA is insufficient"]
    assert canonicalize_mcq("No", opts)[0] is None
    assert canonicalize_mcq("no", opts)[0] is None
    # but the full option text still matches
    assert canonicalize_mcq("John's GPA is insufficient", opts)[0] == "C"


def test_canon_gold_handles_not_given():
    from data_load import _canon_gold
    from schema import AnswerType as AT
    assert _canon_gold("Not Given", AT.YES_NO_UNKNOWN, None) == "Unknown"
    assert _canon_gold("no information available", AT.YES_NO_UNKNOWN, None) == "Unknown"
    assert _canon_gold("No", AT.YES_NO_UNKNOWN, None) == "No"
    assert _canon_gold("Unknown", AT.MCQ, ["a", "b"]) == "Unknown"


def test_8b_load_failure_is_logged():
    rec = Record(id="z", premises_nl=["p"], question_nl="q?",
                 answer_type=AnswerType.YES_NO_UNKNOWN)
    a = query(StubModel("Qwen", lambda u: "Yes"), rec)
    b = query(StubModel("Gemma", lambda u: "No"), rec)
    final = finalize_big_unloadable(rec, a, b, big_model_id="google/gemma-4-E4B-it",
                                    load_error="OOM")
    assert len(final.replies) == 3  # the failed 8B attempt is recorded for the log
    assert "could not be loaded" in final.replies[2].raw
    assert final.answer == "Yes"  # falls back to Qwen's stored answer


def test_parse_reply_answer_first():
    rec = Record(id="x", premises_nl=["p"], question_nl="q?",
                 answer_type=AnswerType.YES_NO_UNKNOWN, options=None)
    canon, display, why = parse_reply("ANSWER: Not Given\nWHY: nothing entails it.", rec)
    assert canon == "Unknown"
    assert "nothing entails it" in why.lower()


def test_cascade_agreement_branch():
    rec = Record(id="agree", premises_nl=["All cats are animals."],
                 question_nl="Are cats animals?", answer_type=AnswerType.YES_NO_UNKNOWN)
    a = StubModel("Qwen", lambda u: "Yes")
    b = StubModel("Gemma", lambda u: "Yes")
    ra, rb = query(a, rec), query(b, rec)
    assert judges_agree(ra, rb)
    final = finalize_agreed(rec, ra, rb)
    assert final.answer == "Yes" and final.agreed
    assert "Qwen" in final.replies[0].model_label  # explanation came from Qwen


def test_cascade_escalation_branch():
    rec = Record(id="dis", premises_nl=["X."], question_nl="Y?",
                 answer_type=AnswerType.YES_NO_UNKNOWN)
    a = StubModel("Qwen", lambda u: "Yes")
    b = StubModel("Gemma", lambda u: "No")
    big = StubModel("Gemma-E4B(8B)", lambda u: "Not Given")
    ra, rb = query(a, rec), query(b, rec)
    assert not judges_agree(ra, rb)
    rc = query(big, rec)
    final = finalize_with_big(rec, ra, rb, rc)
    assert final.answer == "Unknown" and not final.agreed
    assert len(final.replies) == 3  # all three models recorded for the log
