#!/usr/bin/env python
"""Weighted soft-vote runner for the EXACT-style logic dataset.

Idea
----
Up to three stages each answer EVERY question (the bigger models are no longer
fired only on disagreement — they vote on every record). Each model contributes a
weight to the label it picks; the label with the most total weight wins:

    stage "4b"       Qwen/Qwen3.5-4B  +  google/gemma-4-E2B-it   (weight 1.0 each)
    stage "gemma8b"  google/gemma-4-E4B-it                       (weight 1.5)
    stage "liquid8b" LiquidAI/LFM2.5-8B-A1B                      (weight 1.5)

So one 8B (1.5) outvotes a single disagreeing 4B (1.0), but two agreeing 4B
judges (2.0) outvote a lone 8B (1.5). The 8B weight is 1.5x the 4B weight by
default (tune with --weight-4b / --weight-8b).

You choose the line-up with --stages: any comma-combination of {4b, gemma8b,
liquid8b}. Examples: just the two 4B judges (`--stages 4b`), only the Gemma 8B
(`--stages gemma8b`), only Liquid (`--stages liquid8b`), the 4B judges plus the
Gemma 8B (`--stages 4b,gemma8b`), or all three (`--stages 4b,gemma8b,liquid8b`).

Stages run ONE AT A TIME: a stage's model(s) are loaded, vote on every record,
then freed before the next stage loads. So at most {two 4B models} OR {one 8B
model} are resident at once — it fits a 12 GB card. Every model invocation is
logged verbatim to Result/.

Question types: MCQ (pick a letter) and Yes / No / Not Given (the dataset writes
"Not Given" as "Unknown"). The type is decided from the question's structure.

Precision is a single switch applied to every model: 4bit | 8bit | bf16.
NOTE: two 4B models in bf16 will not fit 12 GB — use 4bit (or 8bit) there.

Examples
--------
    # All three stages, 4-bit, scored against gold:
    python run_cascade.py --stages 4b,gemma8b,liquid8b --precision 4bit --show-gold --limit 20

    # The originally-described setup: 4B judges + the Gemma 8B on every record:
    python run_cascade.py --stages 4b,gemma8b --precision 4bit --show-gold

    # Only the Liquid 8B, with reasoning ("thinking") turned on:
    python run_cascade.py --stages liquid8b --precision bf16 --think --show-gold

    # No-GPU wiring smoke test (fake models, exercises every stage + logging):
    python run_cascade.py --backend stub --stages 4b,gemma8b,liquid8b --show-gold --limit 8
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cascade import finalize_by_vote, query  # noqa: E402
from data_load import load_records  # noqa: E402
from logio import (  # noqa: E402
    result_dir, write_model_io, write_predictions_json, write_run_summary,
)
from schema import AnswerType, FinalAnswer, Record  # noqa: E402
from score import score  # noqa: E402

# ── Default model ids ─────────────────────────────────────────────────────────
# These are the repo ids the user specified. If a Gemma download 404s, the
# equivalent current repos are google/gemma-3n-E2B-it / google/gemma-3n-E4B-it —
# override with --gemma-small-model / --gemma-big-model.
DEFAULT_QWEN = "Qwen/Qwen3.5-4B"
DEFAULT_GEMMA_SMALL = "google/gemma-4-E2B-it"
DEFAULT_GEMMA_BIG = "google/gemma-4-E4B-it"
DEFAULT_LIQUID = "LiquidAI/LFM2.5-8B-A1B"
DEFAULT_DATA = ROOT / "Logic_Based_Educational_Queries.json"

# Canonical stage order (4B judges first, then the 8B models). A stage is a unit
# that is loaded together; the "4b" stage holds the two judges, each 8B stage one
# model — which is exactly the VRAM invariant.
STAGE_ORDER = ["4b", "gemma8b", "liquid8b"]
DEFAULT_STAGES = "4b,gemma8b,liquid8b"


def gate(rec: Record) -> tuple[Record, str | None]:
    """Strip gold before inference; re-derive the answer type from structure.
    Returns (model-facing record, gold). `raw` is emptied except for a
    `definitions` list (never gold) so it can't leak the gold answer/explanation
    into the prompt."""
    atype = AnswerType.MCQ if rec.options else AnswerType.YES_NO_UNKNOWN
    safe_raw = {}
    if isinstance(rec.raw, dict) and rec.raw.get("definitions"):
        safe_raw["definitions"] = rec.raw["definitions"]
    gated = Record(
        id=rec.id, premises_nl=rec.premises_nl, question_nl=rec.question_nl,
        answer_type=atype, answer=None, options=rec.options, raw=safe_raw,
    )
    return gated, rec.answer


def parse_stages(spec: str) -> list[str]:
    """'4b , gemma8b' → ['4b', 'gemma8b'] in canonical order, validated/deduped."""
    picked = {tok.strip().lower() for tok in spec.split(",") if tok.strip()}
    bad = picked - set(STAGE_ORDER)
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s) {sorted(bad)}; choose from {STAGE_ORDER} "
            "(comma-separated, e.g. 4b,gemma8b)"
        )
    if not picked:
        raise argparse.ArgumentTypeError("at least one stage is required")
    return [s for s in STAGE_ORDER if s in picked]


def stage_registry(args) -> dict[str, list[dict]]:
    """Per-stage model specs. `stub` is the deterministic answer fn used by the
    no-GPU backend (it exercises agreement, splits, and every weight)."""
    w4, w8 = args.weight_4b, args.weight_8b
    return {
        "4b": [
            dict(id=args.qwen_model, label="Qwen-4B", cls="4b", weight=w4, cot=False,
                 stub=lambda u: "Yes"),
            dict(id=args.gemma_small_model, label="Gemma-E2B", cls="4b", weight=w4, cot=False,
                 stub=lambda u: ("No" if "scholarship" in u.lower() else "Yes")),
        ],
        "gemma8b": [
            dict(id=args.gemma_big_model, label="Gemma-E4B(8B)", cls="8b", weight=w8, cot=False,
                 stub=lambda u: "Not Given"),
        ],
        "liquid8b": [
            dict(id=args.liquid_model, label="LFM2.5-8B-A1B", cls="8b", weight=w8, cot=True,
                 stub=lambda u: "Yes"),
        ],
    }


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
    ap.add_argument("--stages", type=parse_stages, default=DEFAULT_STAGES,
                    help=f"which model groups vote, comma-separated from {STAGE_ORDER} "
                         f"(default: {DEFAULT_STAGES})")
    ap.add_argument("--qwen-model", default=DEFAULT_QWEN, help="4B judge A")
    ap.add_argument("--gemma-small-model", default=DEFAULT_GEMMA_SMALL, help="4B judge B")
    ap.add_argument("--gemma-big-model", default=DEFAULT_GEMMA_BIG, help="the Gemma 8B")
    ap.add_argument("--liquid-model", default=DEFAULT_LIQUID, help="the Liquid 8B (LFM2.5-8B-A1B)")
    ap.add_argument("--weight-4b", type=float, default=1.0, help="vote weight of each 4B model")
    ap.add_argument("--weight-8b", type=float, default=1.5, help="vote weight of each 8B model")
    ap.add_argument("--think", action="store_true",
                    help="enable reasoning/thinking before the answer (Qwen <think>; "
                         "Liquid always reasons; Gemma ignores it). Needs more tokens.")
    ap.add_argument("--backend", choices=["hf", "stub"], default="hf")
    ap.add_argument("--precision", choices=["4bit", "8bit", "bf16", "fp16", "fp32"], default="4bit",
                    help="applied to every model. Two 4B models in bf16 will not fit 12 GB.")
    ap.add_argument("--compute-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="compute dtype for the 4bit/8bit paths (bfloat16 on Blackwell/Ampere)")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="answer budget; auto-raised for thinking / always-reasoning models")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N questions (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="skip the first N questions")
    ap.add_argument("--only", choices=["ynn", "mcq", "all"], default="all")
    ap.add_argument("--show-gold", action="store_true", help="score against the gold answers")
    ap.add_argument("--out", type=Path, default=None, help="extra path to also write predictions JSON")
    args = ap.parse_args()

    stages: list[str] = args.stages  # parse_stages already validated/ordered

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
    golds = [g[1] for g in gated]  # aligned to records by position (ids may recur)

    registry = stage_registry(args)
    print(f"[info] loaded {len(records_all)} questions; processing {len(records)} "
          f"({sum(1 for r in records if r.answer_type == AnswerType.MCQ)} MCQ / "
          f"{sum(1 for r in records if r.answer_type == AnswerType.YES_NO_UNKNOWN)} YNN)")
    print(f"[info] backend={args.backend}  precision={args.precision}  think={args.think}")
    print(f"[info] stages={stages}  weights: 4B={args.weight_4b}  8B={args.weight_8b}")
    for st in stages:
        labels = ", ".join(f"{s['label']} (w={s['weight']:g})" for s in registry[st])
        print(f"[info]   stage '{st}': {labels}")

    def max_tokens_for(model) -> int:
        n = args.max_new_tokens
        if args.think or getattr(model, "always_cot", False):
            n = max(n, 768)
        return n

    def build_stage(specs: list[dict]) -> list:
        """Instantiate (load) a stage's models. The 4B stage builds two models
        that coexist; the 8B stages build one. If the SECOND model of a stage
        fails to load, the first is unloaded before the error propagates — so a
        partial load never leaves a model resident and break the VRAM invariant."""
        def _make(s: dict):
            if args.backend == "stub":
                from chat_model import StubModel
                return StubModel(f"{s['label']}(stub)", s["stub"], vote_weight=s["weight"],
                                 model_class=s["cls"], always_cot=s["cot"])
            from chat_model import ChatModel
            return ChatModel(
                s["id"], precision=args.precision, device_map=args.device_map,
                compute_dtype=args.compute_dtype, enable_thinking=args.think,
                label=s["label"], vote_weight=s["weight"], model_class=s["cls"],
                always_cot=s["cot"])

        models: list = []
        try:
            for s in specs:
                models.append(_make(s))
        except Exception:
            for m in models:  # roll back any already-loaded models in this stage
                try:
                    m.unload()
                except Exception:  # noqa: BLE001
                    pass
            raise
        return models

    # ── Run each stage in turn, collecting every model's vote per record ──────
    # The try/finally GUARANTEES a stage's models are unloaded before the next
    # stage loads, so the VRAM invariant (≤ two 4B OR one 8B resident) always
    # holds. `query()` itself never raises. Votes are accumulated BY POSITION (not
    # by id) so records are never conflated even if two share an id.
    replies_by_pos: list[list] = [[] for _ in records]
    stage_errors: list[tuple[str, str]] = []
    t_start = time.perf_counter()

    for st in stages:
        specs = registry[st]
        print(f"\n[load] stage '{st}': {', '.join(s['label'] for s in specs)}")
        try:
            models = build_stage(specs)
        except Exception as e:  # noqa: BLE001 — a stage that won't load is skipped
            msg = f"{type(e).__name__}: {e}"
            stage_errors.append((st, msg))
            print(f"[error] stage '{st}' failed to load ({msg}); skipping its votes.")
            continue
        try:
            for i, rec in enumerate(records):
                reps = [query(m, rec, max_new_tokens=max_tokens_for(m)) for m in models]
                replies_by_pos[i].extend(reps)
                kind = "MCQ" if rec.answer_type == AnswerType.MCQ else "YNN"
                votes = "  ".join(f"{r.model_label}={r.answer!r}" for r in reps)
                print(f"[{st}][{i + 1:>4}/{len(records)}] {rec.id:<22} {kind}  {votes}")
        finally:
            for m in models:
                m.unload()
            print(f"[unload] stage '{st}' freed")

    # ── Combine votes ─────────────────────────────────────────────────────────
    # A list aligned to `records` by position — never an id-keyed dict, so two
    # records can never collide and silently drop one.
    finals: list[FinalAnswer] = [
        finalize_by_vote(rec, replies_by_pos[i]) for i, rec in enumerate(records)
    ]

    # ── Scoring + logs ────────────────────────────────────────────────────────
    if args.show_gold:
        for f, gold in zip(finals, golds):
            f.gold = gold
    elapsed = time.perf_counter() - t_start
    n_correct = n_scored = 0
    if args.show_gold:
        rep = score(records, finals)
        n_correct, n_scored = rep.overall.correct, rep.overall.total
        print(f"\n[accuracy] {n_correct}/{n_scored} = "
              f"{(n_correct / n_scored if n_scored else 0):.1%}")
        for k, v in rep.by_type.items():
            print(f"           {k:<16} {v.correct}/{v.total} = {v.accuracy:.1%}")
        print(f"[vote]     unanimous: {rep.unanimous}   split: {rep.split}")
    else:
        unan = sum(1 for f in finals if f.agreed)
        print(f"\n[vote]     unanimous: {unan}   split: {len(finals) - unan}")
    if stage_errors:
        for st, msg in stage_errors:
            print(f"[warn] stage '{st}' did not run: {msg}")
    print(f"[done] {len(records)} questions in {elapsed:.1f}s "
          f"({elapsed / max(len(records), 1):.1f}s/q)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = result_dir(ROOT)
    header = {
        "backend": args.backend, "precision": args.precision, "think": args.think,
        "stages": ",".join(stages),
        "weights": f"4B={args.weight_4b} 8B={args.weight_8b}",
        "qwen": args.qwen_model, "gemma_small": args.gemma_small_model,
        "gemma_big": args.gemma_big_model, "liquid": args.liquid_model,
    }
    if stage_errors:
        header["stage_errors"] = "; ".join(f"{st}: {msg}" for st, msg in stage_errors)
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
