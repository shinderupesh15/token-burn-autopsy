# Token Burn Autopsy

**Upload your LLM usage export. Get back one number: how much of last month's
bill was recoverable — itemised, priced, and evidenced.**

Not a cost dashboard. A dashboard shows you what happened; this tells you what
to do. Every finding is priced against published provider rate cards, so each
line reads *"you paid 1.0× where 0.1× was available, on N tokens"* — arithmetic
you can check, not an estimate you have to trust.

---

## The six detectors

| # | Finding | Signal in the log | Lever |
|---|---|---|---|
| 1 | **Cache-miss waste** | Stable prompt prefix recurring with `cache_read_tokens = 0` | Cache reads bill at **0.1×** input (Anthropic) |
| 2 | **Model over-provisioning** | Frontier model + short output + short context | Swap to the tier that fits |
| 3 | **Dead spend** | Tokens billed on errored or `finish_reason: length` calls | Money spent on unusable output |
| 4 | **Runaway agent loop** | Identical `prompt_full_hash` repeating inside one session | Terminate the loop |
| 5 | **Batchable workload** | Off-hours, latency-insensitive traffic on the sync endpoint | Batch API is **50% off** |
| 6 | **Prompt bloat drift** | Mean prompt tokens per agent climbing week-over-week | Prune the system prompt |

Detector thresholds live in `rate_cards.yaml`, not in code — so they can be
challenged and changed live during a demo instead of defended as magic numbers.

---

**52 tests passing.**

## What it reports

On the bundled fixture (150k calls, 6 weeks, an early-stage team):

```
$318 / month recoverable   —  53% of a $601/month bill

CRITICAL  Uncached prompt prefixes                 $146/mo   documented
CRITICAL  Frontier models doing budget-model work  $136/mo   estimated
MEDIUM    Latency-tolerant work on sync endpoint   $49.5/mo  documented
MEDIUM    Spend on calls that produced nothing     $29.0/mo  documented
MEDIUM    System prompts growing week over week    $19.3/mo  estimated
CRITICAL  Runaway agent loops                      $10.3/mo  documented
```

Findings carry a **confidence**: `documented` means the discount is published by
the provider and only the qualifying token count is estimated; `estimated` means
the recommendation needs a human eval before you act on it. Reporting both with
the same authority is how a cost audit loses credibility the first time someone
checks it.

---

## Quickstart

```bash
pip install -r requirements.txt

# generate a 6-week synthetic log (~150k rows)
python src/generate.py --weeks 6 --seed 7 -o data/usage_canonical.csv

# or emit a provider-native shape to exercise the adapters
python src/generate.py --format openai -o data/raw_openai.csv

python -m pytest tests/ -q
```

## Input formats

Auto-detected on upload — **OpenAI**, **Anthropic**, **OpenRouter**,
**Langfuse/LangSmith**, or a file already in canonical form.

Two details make this safe to run on a real bill:

**Prompt text never enters the pipeline.** The schema carries
`prompt_prefix_hash` / `prompt_full_hash` only. If an export does include
prompt text, it is hashed at the adapter boundary and discarded.

**Providers disagree on what "input tokens" means**, and getting it wrong
inflates every savings figure:

- **Anthropic** — `input_tokens` *excludes* cache reads (siblings)
- **OpenAI** — `input_tokens` *includes* cached tokens (a **subset**)

Each adapter declares its convention; the correction happens in exactly one
place, and a test asserts the raw file really did double-count so the test
cannot pass vacuously.

---

## Layout

```
app.py                  Streamlit UI — hero number, loop alarm, findings
rate_cards.yaml         prices + multipliers + thresholds — all editable, no code
src/schema.py           canonical schema; observations only, never conclusions
src/normalize.py        provider adapters -> canonical
src/generate.py         synthetic log; waste emerges, is never labelled
src/rates.py            pricing; unknown models flagged, never zeroed
src/engine.py           the six detectors + de-overlap arithmetic
src/summary.py          template exec summary — no LLM call, no hallucinated figures
tests/test_pipeline.py  asserts the fixture really contains all six patterns
tests/test_engine.py    detector correctness + the money invariants
docs/PROMPT_LOG.md      vibe-coding log, written as the build happened
```

Run it:

```bash
streamlit run app.py
```

## Design rules

1. **Prices are data.** Every dollar figure traces to a line in `rate_cards.yaml`.
2. **The schema stores observations, never conclusions.** No `is_wasteful`
   column. A schema that hands the detector its answer makes the audit circular.
3. **Waste emerges, it is never labelled.** The generator simulates plausible
   engineering behaviour; detecting a pattern you labelled yourself is not
   detection.
4. **Unpriced models are flagged, never silently zeroed.** A model missing from
   the rate card contributing $0 is how a cost audit lies to you.
5. **Not everything is broken.** One agent in the fixture is clean. A report
   that indicts everything reads as fabricated.
6. **Findings overlap; savings do not add.** One nightly job can be both
   over-provisioned and batchable. Savings compose multiplicatively per request
   (`1 - Π(1 - fᵢ)`), never additively — otherwise the report claims more than
   a call ever cost. Four tests enforce this.
