#!/usr/bin/env python
"""Cascade NL-QA runner for the EXACT-style logic dataset.

Idea
----
Two 4B "judges" answer every question using ONLY the natural-language premises
and the question (never the gold answer):

    Qwen/Qwen3.5-4B    +    google/gemma-4-E2B-it      (the two 4B judges)

* AGREE  → that answer stands; Qwen3.5-4B writes the (short) explanation.
* DISAGREE → BOTH 4B models are UNLOADED and the larger model

        google/gemma-4-E4B-it                          (the "8B" tiebreaker)

  is loaded; it decides the answer AND writes the explanation.

At most {two 4B models} OR {one 8B model} are resident at once, so the run fits a
12 GB card. Every model invocation is logged verbatim to Result/.

Question types: MCQ (pick a letter) and Yes / No / Not Given (the dataset writes
"Not Given" as "Unknown"). The type is decided from the question's structure
(lettered options → MCQ, else Yes/No/Not-Given).

Precision is a single switch applied to every model: 4bit | 8bit | bf16.
NOTE: two 4B models in bf16 will not fit 12 GB — use 4bit (or 8bit) there.

Examples
--------
    # Real run (4-bit on a 12 GB card), score against gold:
    python run_cascade.py --precision 4bit --show-gold --limit 20

    # 8-bit judges:
    python run_cascade.py --precision 8bit --show-gold

    # No-GPU wiring smoke test (fake models, exercises both branches + logging):
    python run_cascade.py --backend stub --show-gold --limit 8
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cascade import (  # noqa: E402
    finalize_agreed, finalize_big_unloadable, finalize_with_big, judges_agree, query,
)
from data_load import load_records  # noqa: E402
from logio import (  # noqa: E402
    result_dir, write_model_io, write_predictions_json, write_run_summary,
)
from schema import AnswerType, FinalAnswer, Record  # noqa: E402
from score import score  # noqa: E402

# ── Default model ids ─────────────────────────────────────────────────────────
# These are the repo ids the user specified. If a download 404s, the equivalent
# current repos are google/gemma-3n-E2B-it and google/gemma-3n-E4B-it — override
# with --gemma-small-model / --gemma-big-model (or edit setup.sh's GEMMA_* vars).
DEFAULT_QWEN = "Qwen/Qwen3.5-4B"
DEFAULT_GEMMA_SMALL = "google/gemma-4-E2B-it"
DEFAULT_GEMMA_BIG = "google/gemma-4-E4B-it"
DEFAULT_DATA = ROOT / "Logic_Based_Educational_Queries.json"


def gate(rec: Record) -> tuple[Record, str | None]:
    """Strip gold before inference; re-derive the answer type from structure.
    Returns (model-facing record, gold)."""
    atype = AnswerType.MCQ if rec.options else AnswerType.YES_NO_UNKNOWN
    gated = Record(
        id=rec.id, premises_nl=rec.premises_nl, question_nl=rec.question_nl,
        answer_type=atype, answer=None, options=rec.options, raw={},
    )
    return gated, rec.answer


def _build_stub_models():
    """Two fake 4B judges + one fake 8B. Judge B disagrees on records whose
    question mentions 'scholarship' (just to exercise the escalation path)."""
    from chat_model import StubModel

    def a_fn(_user: str) -> str:
        return "Yes" if "Yes" not in _user else "Yes"

    def b_fn(user: str) -> str:
        return "No" if "scholarship" in user.lower() else "Yes"

    def big_fn(_user: str) -> str:
        return "Not Given"

    return (
        StubModel("Qwen-4B(stub)", a_fn),
        StubModel("Gemma-E2B(stub)", b_fn),
        lambda: StubModel("Gemma-E4B(8B,stub)", big_fn),
    )


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows cp1252 → force UTF-8
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--qwen-model", default=DEFAULT_QWEN, help="4B judge A (and explainer on agreement)")
    ap.add_argument("--gemma-small-model", default=DEFAULT_GEMMA_SMALL, help="4B judge B")
    ap.add_argument("--gemma-big-model", default=DEFAULT_GEMMA_BIG, help="the 8B tiebreaker")
    ap.add_argument("--backend", choices=["hf", "stub"], default="hf")
    ap.add_argument("--precision", choices=["4bit", "8bit", "bf16", "fp16", "fp32"], default="4bit",
                    help="applied to every model. Two 4B models in bf16 will not fit 12 GB.")
    ap.add_argument("--compute-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="compute dtype for the 4bit/8bit paths (bfloat16 on Blackwell/Ampere)")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N questions (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="skip the first N questions")
    ap.add_argument("--only", choices=["ynn", "mcq", "all"], default="all")
    ap.add_argument("--show-gold", action="store_true", help="score against the gold answers")
    ap.add_argument("--out", type=Path, default=None, help="extra path to also write predictions JSON")
    args = ap.parse_args()

    # ── Load + gate ───────────────────────────────────────────────────────────
    records_all = load_records(args.data)
    gated = [gate(r) for r in records_all]
    if args.only == "ynn":
        gated = [g for g in gated if g[0].answer_type == AnswerType.YES_NO_UNKNOWN]
    elif args.only == "mcq":
        gated = [g for g in gated if g[0].answer_type == AnswerType.MCQ]
    gated = gated[args.start:]
    if args.limit:
        gated = gated[: args.limit]
    records = [g[0] for g in gated]
    golds = {g[0].id: g[1] for g in gated}
    print(f"[info] loaded {len(records_all)} questions; processing {len(records)} "
          f"({sum(1 for r in records if r.answer_type == AnswerType.MCQ)} MCQ / "
          f"{sum(1 for r in records if r.answer_type == AnswerType.YES_NO_UNKNOWN)} YNN)")
    print(f"[info] backend={args.backend}  precision={args.precision}")
    print(f"[info] judges: {args.qwen_model}  +  {args.gemma_small_model}")
    print(f"[info] 8B    : {args.gemma_big_model}")

    finals: dict[str, FinalAnswer] = {}
    t_start = time.perf_counter()

    # ── Build the two 4B judges (this is the ONLY time they coexist) ──────────
    if args.backend == "stub":
        model_a, model_b, make_big = _build_stub_models()
    else:
        from chat_model import ChatModel

        def _mk(model_id: str, label: str):
            return ChatModel(model_id, precision=args.precision, device_map=args.device_map,
                             compute_dtype=args.compute_dtype, label=label)

        print("[load] loading the two 4B judges…")
        model_a = _mk(args.qwen_model, "Qwen-4B")
        model_b = _mk(args.gemma_small_model, "Gemma-E2B")
        make_big = lambda: _mk(args.gemma_big_model, "Gemma-E4B(8B)")  # noqa: E731

    # ── Phase 1: both 4B judges answer every record ──────────────────────────
    # The try/finally GUARANTEES both 4B models are unloaded before the 8B is
    # loaded, even if something throws mid-loop — so the VRAM invariant (≤ two 4B
    # OR one 8B resident) holds no matter what. `query()` itself never raises.
    disagreements: list[tuple[Record, object, object]] = []  # (rec, replyA, replyB)
    try:
        for i, rec in enumerate(records, 1):
            rep_a = query(model_a, rec, max_new_tokens=args.max_new_tokens)
            rep_b = query(model_b, rec, max_new_tokens=args.max_new_tokens)
            if judges_agree(rep_a, rep_b):
                finals[rec.id] = finalize_agreed(rec, rep_a, rep_b)
                tag = "agree"
            else:
                disagreements.append((rec, rep_a, rep_b))
                tag = "DISAGREE -> 8B"
            kind = "MCQ" if rec.answer_type == AnswerType.MCQ else "YNN"
            print(f"[1][{i:>4}/{len(records)}] {rec.id:<22} {kind}  "
                  f"A={rep_a.answer!r} B={rep_b.answer!r}  {tag}")
    finally:
        print(f"[unload] freeing the two 4B judges; {len(disagreements)} record(s) need the 8B")
        model_a.unload()
        model_b.unload()

    # ── Phase 2: 8B model decides the disagreements ──────────────────────────
    if disagreements:
        big = None
        big_err: str | None = None
        try:
            print("[load] loading the 8B tiebreaker…")
            big = make_big()
        except Exception as e:  # noqa: BLE001
            big_err = f"{type(e).__name__}: {e}"
            print(f"[error] could not load the 8B model ({big_err}); keeping Qwen's "
                  "answer on disagreements.")
        try:
            for j, (rec, rep_a, rep_b) in enumerate(disagreements, 1):
                if big is None:
                    finals[rec.id] = finalize_big_unloadable(
                        rec, rep_a, rep_b, big_model_id=args.gemma_big_model, load_error=big_err
                    )
                    continue
                rep_c = query(big, rec, max_new_tokens=args.max_new_tokens)
                finals[rec.id] = finalize_with_big(rec, rep_a, rep_b, rep_c)
                kind = "MCQ" if rec.answer_type == AnswerType.MCQ else "YNN"
                print(f"[2][{j:>4}/{len(disagreements)}] {rec.id:<22} {kind}  "
                      f"8B={rep_c.answer!r}  -> {finals[rec.id].answer!r}")
        finally:
            if big is not None:
                print("[unload] freeing the 8B model")
                big.unload()

    # ── Scoring + logs ────────────────────────────────────────────────────────
    if args.show_gold:
        for rid, f in finals.items():
            f.gold = golds.get(rid)
    elapsed = time.perf_counter() - t_start
    n_correct = n_scored = 0
    if args.show_gold:
        rep = score(records, finals)
        n_correct, n_scored = rep.overall.correct, rep.overall.total
        print(f"\n[accuracy] {n_correct}/{n_scored} = "
              f"{(n_correct / n_scored if n_scored else 0):.1%}")
        for k, v in rep.by_type.items():
            print(f"           {k:<16} {v.correct}/{v.total} = {v.accuracy:.1%}")
        print(f"[cascade]  agreed (4B): {rep.agreed}   escalated (8B): {rep.escalated}")
    else:
        agreed = sum(1 for f in finals.values() if f.agreed)
        print(f"\n[cascade]  agreed (4B): {agreed}   escalated (8B): {len(finals) - agreed}")
    print(f"[done] {len(records)} questions in {elapsed:.1f}s "
          f"({elapsed / max(len(records), 1):.1f}s/q)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = result_dir(ROOT)
    header = {
        "backend": args.backend, "precision": args.precision,
        "qwen": args.qwen_model, "gemma_small": args.gemma_small_model,
        "gemma_big": args.gemma_big_model,
    }
    summary = write_run_summary(rd / f"run_cascade_{stamp}.txt", header, records, finals,
                                n_correct, n_scored, elapsed)
    model_io = write_model_io(rd / f"run_cascade_{stamp}_model_io.txt", records, finals)
    preds = write_predictions_json(rd / f"run_cascade_{stamp}.json", records, finals)
    print(f"[wrote] {summary}")
    print(f"[wrote] {model_io}")
    print(f"[wrote] {preds}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_predictions_json(args.out, records, finals)
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
