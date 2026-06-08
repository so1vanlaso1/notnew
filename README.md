# Cascade NL-QA Pipeline

A **two-judge cascade** for the EXACT-style logic dataset
(`Logic_Based_Educational_Queries.json`). It answers each question using **only**
the natural-language premises and the question — never the gold answer, never the
FOL.

## Idea

Two 4B "judges" answer every question:

| Role | Model | Job |
|------|-------|-----|
| Judge A | `Qwen/Qwen3.5-4B` | answer **+** write the explanation on agreement |
| Judge B | `google/gemma-4-E2B-it` | answer |
| Tiebreaker ("8B") | `google/gemma-4-E4B-it` | decide **+** explain on disagreement |

```
                ┌─────────────── phase 1 (two 4B judges loaded) ───────────────┐
  question ───► │  Qwen3.5-4B  ─┐                                               │
                │               ├─► same canonical answer?                      │
  premises ───► │  Gemma-E2B  ──┘        │                                      │
                └───────────────────────┼──────────────────────────────────────┘
                                YES ─────┤───── NO
                                         ▼        ▼
                          answer = agreed     UNLOAD both 4B models
                          why    = Qwen        LOAD Gemma-E4B (the "8B")
                                                        │  phase 2 (only 8B loaded)
                                                        ▼
                                            answer = 8B, why = 8B
```

**VRAM invariant (enforced in code):** at any instant either **two 4B models**
*or* **one 8B model** are resident — never all three. Phase 1 loads the two
judges; they are `unload()`-ed (freeing VRAM) before the 8B is loaded in phase 2.
Each model is loaded once, not per-record.

## Question types

* **MCQ** — pick an option letter `A`–`H` (or `Unknown` if none follows).
* **Yes / No / Not Given** — the dataset writes "Not Given" as **`Unknown`**.
  The prompt shows the model "Not Given"; the output canonicalizes to `Unknown`
  so it lines up with the gold labels.

The type is decided from the question's *structure* (lettered options → MCQ, else
Yes/No/Not-Given), so it never peeks at the gold answer.

## Agreement & explanation

The two judges **agree** only when they produce the *same canonical answer*
(a letter for MCQ; `Yes`/`No`/`Unknown` for the ternary type). If either judge
fails to produce a parseable answer, that counts as a disagreement and the record
escalates to the 8B. Explanations are **short, answer-first** (one sentence) —
from Qwen on agreement, from the 8B on disagreement.

## Precision

One switch, applied to every model: `--precision {4bit,8bit,bf16}`.

> ⚠️ Two 4B models in **bf16** will not fit a 12 GB card. Use **4bit** (≈5–6 GB
> for both judges) or **8bit** there. bf16 is fine if you have the headroom.

## Setup

```bash
cd NEWpipeline
# Gemma repos are gated — accept the license on huggingface.co first:
export HF_TOKEN=hf_xxx
chmod +x setup.sh run.sh
./setup.sh          # GPU check → venv → torch(cu128) → deps → download 3 models
```

`setup.sh` env overrides: `QWEN_ID`, `GEMMA_SMALL_ID`, `GEMMA_BIG_ID`, `CUDA_WHL`,
`HF_TOKEN`. **If a Gemma repo 404s**, the current equivalents are
`google/gemma-3n-E2B-it` / `google/gemma-3n-E4B-it` — re-run with
`GEMMA_SMALL_ID=…  GEMMA_BIG_ID=…  ./setup.sh`.

## Run

```bash
# real run, 4-bit, scored against gold:
python run_cascade.py --precision 4bit --show-gold --limit 20
# 8-bit, MCQ only:
python run_cascade.py --precision 8bit --only mcq --show-gold
# whole dataset:
python run_cascade.py --precision 4bit --show-gold
# no-GPU wiring test (fake models, exercises both branches + logging):
python run_cascade.py --backend stub --show-gold --limit 8
```

Useful flags: `--limit N`, `--start N`, `--only {ynn,mcq,all}`, `--max-new-tokens`,
`--qwen-model/--gemma-small-model/--gemma-big-model` (override repo ids),
`--out path.json`.

## Output / logs

Every run drops three timestamped files in `Result/` (mirrors the old pipeline):

| File | Contents |
|------|----------|
| `run_cascade_<stamp>.txt` | per-record summary: question, both judges' votes, agreement, 8B vote (if fired), final answer + confidence + short WHY, gold |
| `run_cascade_<stamp>_model_io.txt` | **every model invocation, verbatim** — the exact prompt in and raw text out, for both judges and the 8B |
| `run_cascade_<stamp>.json` | machine-readable predictions (answer, explanation, votes, decider, confidence) |

## Layout

```
run_cascade.py        entry point — two-phase load/unload orchestration + logging
src/
  schema.py           Record / ModelReply / FinalAnswer (plain dataclasses)
  data_load.py        dataset → Record (MCQ split, gold canonicalization)
  prompts.py          prompt builders + answer parsing/normalization
  chat_model.py       ChatModel (4bit/8bit/bf16, load/unload) + StubModel
  cascade.py          agreement check + finalize-on-agree / finalize-with-8B
  logio.py            the three Result/ writers
  score.py            accuracy + per-type + agree/escalate counts
tests/test_smoke.py   no-GPU tests (data, normalization, both cascade branches)
configs/default.yaml  documented defaults
setup.sh / run.sh     quickstart + convenience wrapper
```

## Notes

* The "4 models" in the original brief are really **3 distinct models** (two 4B
  judges + one larger tiebreaker); the agreement step is the cross-model "vote".
* Decoding is greedy (`temperature=0`) so a model's answer is reproducible — the
  "vote" is across the two models, not across samples of one model.
* Only `premises-NL` + `questions` are read from the dataset; `premises-FOL` and
  the gold `answers`/`explanation` are never shown to a model.
