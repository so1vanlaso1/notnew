"""No-GPU smoke tests: data loading, prompt/answer normalization, and the
weighted soft-vote wiring via StubModel (agreement, splits, and weights).

Run:  python -m pytest NEWpipeline/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cascade import (  # noqa: E402
    WEIGHT_4B, WEIGHT_8B, finalize_by_vote, query, tally,
)
from chat_model import StubModel  # noqa: E402
from data_load import load_records, parse_mcq_question  # noqa: E402
from prompts import (  # noqa: E402
    build_user, canonicalize_mcq, canonicalize_ynn, parse_reply, strip_thinking,
)
from run_cascade import STAGE_ORDER, parse_stages  # noqa: E402
from schema import AnswerType, ModelReply, Record  # noqa: E402

DATA = ROOT / "Logic_Based_Educational_Queries.json"


# ── helpers ───────────────────────────────────────────────────────────────────
def _ynn(rid: str = "r") -> Record:
    return Record(id=rid, premises_nl=["All cats are animals."],
                  question_nl="Are cats animals?", answer_type=AnswerType.YES_NO_UNKNOWN)


def _reply(answer, weight, cls="4b", why="because premise 1") -> ModelReply:
    return ModelReply(model_label=f"m@{weight}", model_id="stub", prompt="", raw="",
                      answer=answer, answer_display=(answer or ""), explanation=why,
                      weight=weight, model_class=cls)


# ── data / normalization (unchanged behavior) ─────────────────────────────────
def test_dataset_loads_and_classifies():
    recs = load_records(DATA)
    assert len(recs) > 500
    kinds = {r.answer_type for r in recs}
    assert AnswerType.MCQ in kinds and AnswerType.YES_NO_UNKNOWN in kinds
    for r in recs:
        if r.answer is None:
            continue
        if r.answer_type == AnswerType.MCQ:
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
    opts = ["Sophia is eligible for the program",
            "Sophia needs to pass to get an honors diploma",
            "John's GPA is insufficient"]
    assert canonicalize_mcq("No", opts)[0] is None
    assert canonicalize_mcq("no", opts)[0] is None
    assert canonicalize_mcq("John's GPA is insufficient", opts)[0] == "C"


def test_canon_gold_handles_not_given():
    from data_load import _canon_gold
    from schema import AnswerType as AT
    assert _canon_gold("Not Given", AT.YES_NO_UNKNOWN, None) == "Unknown"
    assert _canon_gold("no information available", AT.YES_NO_UNKNOWN, None) == "Unknown"
    assert _canon_gold("No", AT.YES_NO_UNKNOWN, None) == "No"
    assert _canon_gold("Unknown", AT.MCQ, ["a", "b"]) == "Unknown"


# ── new prompt template ───────────────────────────────────────────────────────
def test_build_user_ynn_has_statement():
    u = build_user(_ynn())
    assert "Premises:" in u and "Definitions: None" in u
    assert "Question:" in u and "Statement: Are cats animals?" in u
    assert "Options:" not in u


def test_build_user_mcq_has_options_not_statement():
    rec = Record(id="m", premises_nl=["p1", "p2"], question_nl="Which follows?",
                 answer_type=AnswerType.MCQ, options=["first", "second"])
    u = build_user(rec)
    assert "Question: Which follows?" in u
    assert "Options:" in u and "A. first" in u and "B. second" in u
    assert "Statement:" not in u
    # premises are numbered from 1 so the rules can cite "premise 2"
    assert "1. p1" in u and "2. p2" in u


# ── parsing: inline WHY + thinking strip ──────────────────────────────────────
def test_parse_reply_answer_first():
    rec = _ynn()
    canon, display, why = parse_reply("ANSWER: Not Given\nWHY: nothing entails it.", rec)
    assert canon == "Unknown"
    assert "nothing entails it" in why.lower()


def test_parse_reply_inline_answer_and_why():
    rec = _ynn()
    canon, display, why = parse_reply("ANSWER: Yes WHY: premise 1 entails it.", rec)
    assert canon == "Yes"
    assert display == "Yes"  # the inline WHY is not glued onto the answer token
    assert "premise 1" in why.lower()


def test_parse_reply_strips_thinking_block():
    rec = _ynn()
    raw = "<think>It might be No, but actually...</think>\nANSWER: Yes WHY: by premise 1."
    canon, _disp, why = parse_reply(raw, rec)
    assert canon == "Yes"  # the 'No' inside <think> must not win
    assert "<think>" not in why and "It might be No" not in why


def test_strip_thinking_handles_unclosed_close_tag():
    assert "reasoning" not in strip_thinking("reasoning here</think> ANSWER: Yes").lower()


# ── weighted soft vote ────────────────────────────────────────────────────────
def test_weights_defaults():
    assert WEIGHT_4B == 1.0 and WEIGHT_8B == 1.5


def test_tally_sums_weights():
    s = tally([_reply("Yes", 1.0), _reply("Yes", 1.0), _reply("No", 1.5, "8b")])
    assert s == {"Yes": 2.0, "No": 1.5}


def test_one_8b_outvotes_single_4b():
    rec = _ynn()
    final = finalize_by_vote(rec, [_reply("Yes", 1.0), _reply("No", 1.5, "8b", why="8B says no")])
    assert final.answer == "No" and not final.agreed
    assert abs(final.confidence - 1.5 / 2.5) < 1e-9
    assert final.explanation == "8B says no"  # winner's explanation is the 8B's


def test_two_4b_outvote_one_8b():
    rec = _ynn()
    final = finalize_by_vote(
        rec, [_reply("Yes", 1.0), _reply("Yes", 1.0), _reply("No", 1.5, "8b")])
    assert final.answer == "Yes"
    assert abs(final.confidence - 2.0 / 3.5) < 1e-9


def test_unanimous_is_flagged_and_full_confidence():
    rec = _ynn()
    final = finalize_by_vote(rec, [_reply("Yes", 1.0), _reply("Yes", 1.5, "8b")])
    assert final.answer == "Yes" and final.agreed
    assert abs(final.confidence - 1.0) < 1e-9


def test_pure_4b_tie_is_broken_deterministically():
    rec = _ynn()
    final = finalize_by_vote(rec, [_reply("Yes", 1.0), _reply("No", 1.0)])
    # tie on weight → defer to the later (more authoritative) vote
    assert final.answer == "No" and not final.agreed
    assert abs(final.confidence - 0.5) < 1e-9


def test_explanation_prefers_strongest_model_with_a_reason():
    rec = _ynn()
    final = finalize_by_vote(rec, [
        _reply("Yes", 1.0, why="4B reason"),
        _reply("Yes", 1.5, "8b", why=""),       # strongest, but no WHY
        _reply("Yes", 1.0, why="later 4B reason"),
    ])
    # Skips the empty-WHY 8B; among equal-weight explainers, prefers the later one
    # (same rule as the vote tie-break).
    assert final.explanation == "later 4B reason"


def test_explanation_uses_8b_when_it_explains():
    rec = _ynn()
    final = finalize_by_vote(rec, [
        _reply("Yes", 1.0, why="4B reason"),
        _reply("Yes", 1.5, "8b", why="8B reason"),
    ])
    assert final.explanation == "8B reason"  # strongest model that explained wins


def test_no_parseable_answer_yields_none():
    rec = _ynn()
    final = finalize_by_vote(rec, [_reply(None, 1.0), _reply(None, 1.5, "8b")])
    assert final.answer is None and final.confidence == 0.0
    assert not final.agreed


# ── end-to-end via StubModel.query ────────────────────────────────────────────
def test_query_copies_weight_and_class_onto_reply():
    rec = _ynn()
    m = StubModel("Big", lambda u: "No", vote_weight=1.5, model_class="8b")
    rep = query(m, rec)
    assert rep.answer == "No" and rep.weight == 1.5 and rep.model_class == "8b"


def test_full_three_stage_vote_via_stubs():
    rec = Record(id="dis", premises_nl=["X."], question_nl="Does she get a scholarship?",
                 answer_type=AnswerType.YES_NO_UNKNOWN)
    qwen = StubModel("Qwen-4B", lambda u: "Yes", vote_weight=1.0, model_class="4b")
    gemma_s = StubModel("Gemma-E2B", lambda u: "No", vote_weight=1.0, model_class="4b")
    gemma_b = StubModel("Gemma-E4B", lambda u: "Not Given", vote_weight=1.5, model_class="8b")
    liquid = StubModel("Liquid", lambda u: "Yes", vote_weight=1.5, model_class="8b")
    reps = [query(m, rec) for m in (qwen, gemma_s, gemma_b, liquid)]
    final = finalize_by_vote(rec, reps)
    # Yes = 1.0(qwen) + 1.5(liquid) = 2.5 ; No = 1.0 ; Unknown = 1.5
    assert final.scores == {"Yes": 2.5, "No": 1.0, "Unknown": 1.5}
    assert final.answer == "Yes"
    assert len(final.replies) == 4  # all four models recorded for the log


# ── positional alignment is robust to duplicate ids ───────────────────────────
def test_predictions_dict_keeps_records_with_a_shared_id():
    from logio import predictions_dict
    rec = _ynn("dup")
    f1 = finalize_by_vote(rec, [_reply("Yes", 1.0)])
    f2 = finalize_by_vote(rec, [_reply("No", 1.0)])
    out = predictions_dict([rec, rec], [f1, f2])
    assert len(out) == 2  # both survive even though the records share an id
    assert sorted(v["answer"] for v in out.values()) == ["No", "Yes"]


def test_score_aligns_by_position_not_id():
    from score import score
    r1 = Record(id="x", premises_nl=["p"], question_nl="q1?",
                answer_type=AnswerType.YES_NO_UNKNOWN, answer="Yes")
    r2 = Record(id="x", premises_nl=["p"], question_nl="q2?",
                answer_type=AnswerType.YES_NO_UNKNOWN, answer="No")
    rep = score([r1, r2], [finalize_by_vote(r1, [_reply("Yes", 1.0)]),
                           finalize_by_vote(r2, [_reply("No", 1.0)])])
    assert rep.overall.correct == 2  # both scored independently, not collapsed


# ── stage selection ───────────────────────────────────────────────────────────
def test_parse_stages_orders_and_dedupes():
    assert parse_stages("liquid8b,4b,4b") == ["4b", "liquid8b"]
    assert parse_stages("4b,gemma8b,liquid8b") == STAGE_ORDER


def test_parse_stages_rejects_unknown():
    import pytest
    with pytest.raises(Exception):
        parse_stages("4b,bogus")
