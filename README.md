# Cascade NL-QA Pipeline

A **weighted soft-vote** ensemble for the EXACT-style logic dataset
(`Logic_Based_Educational_Queries.json`). It answers each question using **only**
the natural-language premises and the question — never the gold answer, never the
FOL.

## Idea

Up to three **stages** each answer **every** question, and the answers are
combined by a weighted vote. The bigger models are no longer fired only on
disagreement — they vote on every record.

| Stage | Model(s) | Class | Vote weight |
|-------|----------|-------|-------------|
| `4b` | `Qwen/Qwen3.5-4B` **+** `google/gemma-4-E2B-it` | 4B | **1.0** each |
| `gemma8b` | `google/gemma-4-E4B-it` | 8B | **1.5** |
| `liquid8b` | `LiquidAI/LFM2.5-8B-A1B` (MoE, 8.3B/1.5B active) | 8B | **1.5** |

Each model adds its weight to the label it picked; the label with the most total
weight wins. So **one 8B (1.5) outvotes a single disagreeing 4B (1.0)**, but
**two agreeing 4B judges (2.0) outvote a lone 8B (1.5)**. Confidence is the
winning label's share of the total weight, and the final explanation comes from
the strongest model that voted for the winning label (an 8B's reasoning beats a
4B's). The 8B/4B weight ratio (default 1.5x) is tunable with `--weight-4b` /
`--weight-8b`.

```
  ┌── stage 4b (two 4B judges loaded together) ──┐
  │  Qwen3.5-4B   → vote (w 1.0)                  │   then UNLOAD, then
  │  Gemma-E2B    → vote (w 1.0)                  │   ┌── stage gemma8b ──┐   ┌── stage liquid8b ──┐
  └──────────────────────────────────────────────┘   │ Gemma-E4B → 1.5   │   │ LFM2.5-8B → 1.5     │
                                                      └───────────────────┘   └─────────────────────┘
                         every vote, per record  ─────────────►  weighted tally  ─►  argmax = answer
```

## Choosing the line-up (`--stages`)

You pick which stages run — any comma-combination of `{4b, gemma8b, liquid8b}`:

```bash
--stages 4b                    # just the two 4B judges
--stages gemma8b               # only the Gemma 8B
--stages liquid8b              # only the Liquid 8B
--stages 4b,gemma8b            # 4B judges + Gemma 8B   (the "call the 8B every run" setup)
--stages 4b,gemma8b,liquid8b   # all three (default)
```

**VRAM invariant (enforced in code):** stages run **one at a time** — a stage's
model(s) load, vote on every record, then are `unload()`-ed (freeing VRAM)
before the next stage loads. So at any instant either **two 4B models** *or*
**one 8B model** are resident — never more. Each model is loaded once, not
per-record. A stage that fails to load (e.g. wrong `transformers` for Liquid) is
logged and **skipped**; the remaining stages still vote.

## Question types

* **MCQ** — pick an option letter `A`–`H` (or `Unknown` if none follows).
* **Yes / No / Not Given** — the dataset writes "Not Given" as **`Unknown`**.

The type is decided from the question's *structure* (lettered options → MCQ, else
Yes/No/Not-Given), so it never peeks at the gold answer.

## Prompts

The system prompts are **strict formal-logic examiner** rules that target the
exact mistakes the 4B judges kept making — reading "not enough information" as
**No** (lack of proof ⇒ *Not Given*, not *No*), the **converse**/**inverse**
fallacies, and confusing **some** with **all**. The user turn follows a fixed
template:

```
Premises:        (numbered from 1, so the model can cite "premise 7")
Definitions:     None        (the dataset folds definitions into the premises)
Question:        the decision task (YNN) / the stem (MCQ)
Statement:       the claim to test          (YNN only)
Options:         A. … B. …                  (MCQ only)
```

Each model replies `ANSWER: <…> WHY: <one short sentence citing premise numbers>`.

## Thinking mode

`--think` turns on reasoning before the answer: Qwen reasons inside
`<think>…</think>`, Gemma ignores the flag, and Liquid always reasons. The
`<think>` block (or any chain-of-thought) is stripped before the `ANSWER:` line
is parsed, and the token budget is auto-raised to ≥768 so the answer isn't cut
off.

## Precision

One switch, applied to every model: `--precision {4bit,8bit,bf16}`.

> ⚠️ Two 4B models in **bf16** will not fit a 12 GB card. Use **4bit** (≈5–6 GB
> for both judges) or **8bit** there. bf16 is fine if you have the headroom.

On a **single GPU**, the loader pins each model fully to GPU 0 (it does **not**
let `device_map="auto"` offload to CPU/disk, which would otherwise make
bitsandbytes int8 refuse to load an 8B model that actually fits). If a model is
genuinely too big it OOMs cleanly and that stage is skipped. The Liquid 8B MoE
fits at **8bit** (~8–9 GB) or **4bit** (~4.5 GB) on a 12 GB card — drop to 4bit
if 8bit is tight because other processes hold VRAM.

## Setup

```bash
cd NEWpipeline
# Gemma repos are gated — accept the license on huggingface.co first:
export HF_TOKEN=hf_xxx
chmod +x setup.sh run.sh
./setup.sh          # GPU check → venv → torch(cu128) → deps → download 4 models
```

`setup.sh` env overrides: `QWEN_ID`, `GEMMA_SMALL_ID`, `GEMMA_BIG_ID`,
`LIQUID_ID`, `CUDA_WHL`, `HF_TOKEN`. **If a Gemma repo 404s**, the current
equivalents are `google/gemma-3n-E2B-it` / `google/gemma-3n-E4B-it`. **Liquid**
(`LiquidAI/LFM2.5-8B-A1B`) is **not gated** but needs **`transformers>=5.0`**
(pinned in `requirements.txt`). You only need the models for the stages you run.

## Run

```bash
# all three stages, 4-bit, scored against gold:
python run_cascade.py --stages 4b,gemma8b,liquid8b --precision 4bit --show-gold --limit 20
# 4B judges + Gemma 8B on every record:
python run_cascade.py --stages 4b,gemma8b --precision 4bit --show-gold
# only the Liquid 8B, with thinking on:
python run_cascade.py --stages liquid8b --precision bf16 --think --show-gold
# no-GPU wiring test (fake models, exercises every stage + logging):
python run_cascade.py --backend stub --stages 4b,gemma8b,liquid8b --show-gold --limit 8
```

Useful flags: `--stages`, `--weight-4b/--weight-8b`, `--think`, `--limit N`,
`--start N`, `--only {ynn,mcq,all}`, `--max-new-tokens`, `--out path.json`,
`--qwen-model/--gemma-small-model/--gemma-big-model/--liquid-model` (override
repo ids).

## Output / logs

Every run drops three timestamped files in `Result/`:

| File | Contents |
|------|----------|
| `run_cascade_<stamp>.txt` | per-record summary: **every model's weighted vote**, the tally, the winning answer + confidence + short WHY, gold |
| `run_cascade_<stamp>_model_io.txt` | **every model invocation, verbatim** — the exact prompt in and raw text out, with each model's weight/class |
| `run_cascade_<stamp>.json` | machine-readable predictions (answer, explanation, per-label `scores`, votes with weights, confidence) |

## Layout

```
run_cascade.py        entry point — per-stage load/vote/unload + weighted vote + logging
src/
  schema.py           Record / ModelReply (+weight/class) / FinalAnswer (+scores)
  data_load.py        dataset → Record (MCQ split, gold canonicalization, unique ids)
  prompts.py          strict-logic prompts + answer parsing (<think> stripping)
  chat_model.py       ChatModel (4bit/8bit/bf16, thinking, load/unload) + StubModel
  cascade.py          query() + weighted_vote / finalize_by_vote
  logio.py            the three Result/ writers
  score.py            accuracy + per-type + unanimous/split counts
tests/test_smoke.py   no-GPU tests (data, normalization, prompts, weighted vote, stages)
configs/default.yaml  documented defaults
setup.sh / run.sh     quickstart + convenience wrapper
```

## Notes

* Decoding is greedy (`temperature=0`) so each model's answer is reproducible —
  the "vote" is across models, not across samples of one model.
* Only `premises-NL` + `questions` are read from the dataset; `premises-FOL` and
  the gold `answers`/`explanation` are never shown to a model.
* Record ids are made globally unique (`r<pos>_<idx>_q<n>`) because the dataset's
  `idx` is a list of premise indices, not a unique key, and recurs across records.
```
