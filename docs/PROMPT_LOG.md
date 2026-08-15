# Vibe-coding log — Token Burn Autopsy

> A log of the prompts used, the iterations tried, and the learnings along the
> way. This file is written **as the build happens**, not reconstructed
> afterwards. The dead ends are kept in on purpose — a log with no wrong turns
> in it is a log that was written at the end.

---

## Part 1 — schema, adapters, data generator

### Iteration 1 — framing the problem before writing any code

The first decision was not a prompt, it was a rejection. The obvious build here
is an "LLM cost dashboard": spend over time, tokens by model, a date filter.
That gets built by everyone and says nothing.

The reframe that survived: **this is not a dashboard, it is an auditor.** It
takes a usage export and returns one number — money recoverable this month —
itemised into findings you can act on today. A dashboard shows you what
happened. An auditor tells you what to do.

Everything below follows from that framing. The hero output is a savings
ledger, not a chart.

### Iteration 2 — "make the pricing data, not code"

Rather than hardcoding prices in Python, prices went into `rate_cards.yaml`.
The reason is defensive: in a live demo someone will ask *"where does that
$ figure come from?"* and the answer needs to be a file you can open, not a
constant buried in a function.

The YAML separates two kinds of number, which turned out to matter more than
expected:

| Kind | Confidence | Example |
|---|---|---|
| **Multipliers** | Provider-documented, stable | cache read = `0.1x` base input (Anthropic) |
| **Base prices** | Change often, user-editable | `claude-opus-5` input = `$5.00/MTok` |

Verified while building, and now the backbone of the savings engine:

- Anthropic cache **reads cost 0.1×** base input; 5-min cache **writes 1.25×**,
  1-hour writes **2×**
- **Batch API is 50% off** on both major providers

So a finding is never a vibe. It is always *"you paid 1.0× where 0.1× was
available, on N tokens."*

**Non-obvious catch:** OpenAI's cached-input discount is nowhere near
Anthropic's `0.1×`. Assuming one number across providers would have
overstated savings on every OpenAI row. The rate card now stores the
multiplier per provider, and the asymmetry is itself worth surfacing in the
report.

### Iteration 3 — schema first, and one rule that shaped it

Prompt intent: *"design the canonical schema every provider export normalises
onto."*

The rule that ended up governing `schema.py`:

> **Store observations, never conclusions.**

There is no `is_wasteful` column, no `should_cache` flag. If a detector finds
the cache-miss pattern it has to derive it from raw token counts. A schema that
hands the detector its answer makes the audit circular, and circular audits are
exactly the "AI slop" this project is trying not to be.

Second decision: fingerprints, **not prompt text**. The schema carries
`prompt_prefix_hash` and `prompt_full_hash`, never the prompt itself. That is
what makes it safe to run the tool on a real production log — which is the
whole difference between a demo and something a viewer would actually try on
their own bill.

### Iteration 4 — the bug that justified the whole adapter layer

Building the provider adapters surfaced something I had not anticipated, and it
would have quietly corrupted every dollar figure in the report:

> **Providers disagree about what "input tokens" means.**
>
> - **Anthropic**: `input_tokens` *excludes* `cache_read_input_tokens` — siblings.
> - **OpenAI**: `input_tokens` *includes* `cached_tokens` — a **subset**.

Normalise naively and OpenAI's cached tokens get counted twice: once as input,
once as cache. Every "you could save $X by caching" number inflates.

Fix: each adapter declares `cache_is_subset`, and the correction happens in
exactly one place. `test_openai_adapter_subtracts_cached_from_input` asserts
both that the adapter fixes it *and* that the raw file really did double-count
— otherwise the test proves nothing.

### Iteration 5 — generator: let waste emerge, don't label it

Prompt intent: *"generate a believable 6-week usage log containing all six
waste patterns."*

Two rules kept this honest:

1. **The generator never writes a waste flag.** It simulates plausible
   engineering behaviour and lets waste *emerge* from token counts. Detecting
   a pattern you yourself labelled is not detection.
2. **Not everything is broken.** `invoice_parser` caches properly, runs a
   budget model, uses the batch endpoint. A report that indicts all six agents
   reads as fabricated; the clean control is what makes the guilty ones legible.

Scenario: "Northwind", six weeks into shipping LLM features, with the cost
mistakes a team makes in its first quarter.

| Agent | Seeded behaviour | Pattern it produces |
|---|---|---|
| `support_triage` | 9-way classifier, ~30 output tokens, on a frontier model | Model over-provisioning |
| `doc_summariser` | 6.4k-token style guide re-sent every call, caching off | Cache-miss waste |
| `research_v2` | Multi-step agent; ~5% of runs hit a non-terminating tool | Runaway loop |
| `nightly_enrichment` | 1am backfill on the real-time endpoint | Batchable workload |
| `chat_api` | System prompt grows ~19%/week, never pruned | Prompt bloat drift |
| *all* | Rate limits, overloads, truncated generations | Dead spend |
| `invoice_parser` | **Cached, batched, right-sized** | *(clean control)* |

### Iteration 6 — two real bugs, caught by probing the output

This is the part that best shows what vibe coding actually looks like. The generator ran, produced 149,982 rows, and looked fine.
Probing the output showed it was not.

**Bug 1 — the clean control had a 0% cache hit rate.**

`invoice_parser` was supposed to be the well-behaved agent, but reported zero
cache reads. Cause: cache hits were modelled as *position within a session* —
first call writes, later calls read. `invoice_parser` has single-call sessions,
so every call was a write and nothing ever hit.

The model was simply wrong about how prompt caching works. **A cache entry
lives on a TTL, not a session.** A hit depends on wall-clock time since that
prefix was last touched — across sessions, across users.

Rewrote it as `_apply_cache_semantics()`: walk the merged timeline once, keyed
on `(agent, model, prefix)`; within the 5-minute TTL it is a read, otherwise a
write. This required deferring cache assignment until *after* all agents merge
into one sorted timeline, since a hit can depend on a different session's call.

Hit rate went 0% → **96.7%**, and the control agent finally looks clean.
Locked in by `test_cache_hits_follow_ttl_not_session`.

**Bug 2 — prompt bloat was invisible.**

`chat_api`'s prefix was configured to grow 19%/week, but mean `input_tokens`
per week came out flat: `[300, 299, 301, 299, 299, 296]`.

Not a generator bug — a **detector design** finding. `chat_api` *caches*, so
its growing prefix lands in `cache_read_tokens`, not `input_tokens`. Measured
on total prompt tokens instead:

```
1500 → 1727 → 2000 → 2321 → 2705 → 3159    (+111% over 6 weeks)
```

The learning generalises: **a drift detector reading `input_tokens` alone will
silently miss bloat in exactly the agents that cache** — the mature ones, where
prefixes are largest. `test_pattern_6_prompt_bloat_drift` asserts both that
total tokens climb *and* that `input_tokens` stays flat, so the wrong
implementation cannot pass.

### Iteration 7 — tests as the contract for Part 2

17 tests, all passing. The suite asserts the fixture provably contains each of
the six patterns *before* any detector exists. Without that, the detectors
built next would be validated against data nobody had verified — and a
detector that "finds" a pattern in unverified data proves nothing.

```
$ python -m pytest tests/ -q
17 passed in 19.23s
```

---

## Observations on the workflow so far

**Where AI assistance paid off most:** the boilerplate-heavy, well-specified
layers — schema definition, four provider adapters, the fixture generator.
Hundreds of lines that are tedious but not conceptually hard.

**Where it needed steering:** the framing (a "cost dashboard" is the default
suggestion and it is the wrong product), and both bugs above. Neither bug was a
crash. The code ran, produced plausible-looking output, and was wrong in a way
only visible by probing the result against what it was *supposed* to contain.

**The actual lesson:** generated code fails silently far more often than it
fails loudly. The habit that caught both bugs was writing a probe that asks
"is the thing I intended actually in here?" immediately after generating —
before building anything on top of it. Speed came from generation; correctness
came from verification, and those are separate activities.

---

## Part 2 — savings engine + Streamlit UI

### Iteration 8 — pricing, and one rule that shaped `rates.py`

The pricing layer holds a single opinion firmly:

> **A model missing from the rate card is not priced at zero. It is flagged and
> excluded, and the exclusion is reported.**

A cost audit that silently prices unknown rows at $0 understates the bill and
tells you the *opposite* of the truth. The fixture deliberately contains 224
calls on a `gpt-4o-legacy` model that is absent from `rate_cards.yaml`, so the
Data Quality tab always has something real to show.

### Iteration 9 — the overlap bug (the one that would have cost the most)

The first end-to-end run looked great: **$555 recoverable on an $845 bill.**
Then the per-finding evidence tables were printed side by side:

```
right_sizing   nightly_enrichment   gpt-5 → gpt-5-mini   paid $96.37   recoverable $77.10
batchable      nightly_enrichment   off-hours 100%       paid $136.90  recoverable $68.45
```

`nightly_enrichment` cost **$136.90** in total, and the report was claiming
**$145.55** of savings on it. Over 100%.

**Findings are not disjoint.** One nightly job is simultaneously over-provisioned
*and* batchable. Adding the two claims money that does not exist, and anyone
who divides one column by another finds it in about fifteen seconds.

The fix was a real refactor, not a patch. Every detector now returns
`row_savings` — a per-`request_id` series, not just a group total — and
`_combine()` composes them the way the levers actually compose on a single call:

```python
combined_fraction = 1 - Π(1 - fraction_i)
```

Batching at 0.5× on top of a model swap saves half of the *already-reduced*
price, not half of the original. This is bounded by the row's real cost no
matter how many findings name it.

| | |
|---|---|
| Naive sum of findings | $555 |
| **De-overlapped total** | **$453** |
| Overlap correction | $102 |

Rather than hide this, the app *states* it — the exec summary explains why the
findings below it do not add up to the headline. It reads as rigour instead of
an inconsistency someone else gets to find.

Four invariants now make the bug unrepeatable:
`test_recoverable_never_exceeds_spend`, `test_no_agent_is_claimed_beyond_its_own_spend`,
`test_no_single_request_is_saved_twice`, `test_leaderboard_sums_to_the_headline`.

### Iteration 10 — the cache detector was under-reporting its biggest target

With overlap fixed, something still looked wrong. `doc_summariser` re-sends a
**7,088-token** prefix across 9,209 calls — by far the largest waste in the
fixture — but the detector ranked `support_triage` (1,055 tokens) above it.

The cause was the TTL. The detector priced caching as one write per **5-minute**
window. `doc_summariser` runs ~260 calls spread over an 11-hour day, so its
calls land minutes apart: nearly every call opens a *new* window, pays the 1.25×
write premium, and almost never gets a read. Correctly computed, and correctly
useless.

But that is an argument for a **different TTL**, not against caching. The
1-hour window costs 2× to write and turns those same sparse calls into reads at
0.1×. The detector now evaluates both and reports whichever is cheaper, with a
`best_ttl` column in the evidence.

```
cache-miss finding:  $75  →  $209
doc_summariser correctly becomes the top cost criminal
```

The general lesson: **the first version was arithmetically right and
analytically wrong.** It answered "would the default TTL help?" when the
question was "would caching help?" Detectors that evaluate only the default
configuration systematically under-report on exactly the workloads that need
the most help.

### Iteration 11 — the exec summary: template, not an LLM call

Deliberate, for three reasons:

1. A cost auditor that spends tokens to tell you about your token spend is a
   bad joke waiting to happen.
2. Every sentence is generated from a computed number. A model would
   *paraphrase* those numbers, and paraphrasing is where a hallucinated figure
   enters a financial report.
3. It runs offline with no key, so the demo cannot fail live.

### Iteration 12 — two UI bugs only visible by looking

Both found by screenshotting the running app rather than trusting that it ran.

**The summary rendered as an equation.** Streamlit's markdown has LaTeX
enabled, so `cost $845 ($604/month...` — two dollar signs on one line — was
swallowed as math and rendered in italic serif with the `$` stripped. Escaping
to `\$` fixes it. Also worth recording: **Streamlit does not hot-reload
imported modules**, so the first fix appeared to do nothing and needed a server
restart to verify — a false negative that nearly sent me looking in the wrong
place.

**Bar labels sat on the segment boundary.** Value labels with
`textposition="outside"` on a *stacked* bar land at the end of the blue
segment, which is the middle of the stack — so the number read as if it
belonged to the grey. Moved to annotations past the end of the whole stack,
with the share appended (`$153 · 53%`).

### Iteration 13 — charts

Built against a validated palette: one categorical hue for "recoverable",
a deliberately recessive grey for "irreducible spend", status colours reserved
for severity and never reused as a series. No dual axes, no pie charts, no
value label on every point. `invoice_parser` renders as an almost entirely grey
bar — the visual proof that the tool is not just indicting everything it sees.

**48 tests passing.**

---

## Observations after Part 2

Three bugs across the build. Not one of them crashed. Every one produced
plausible output that a reasonable person would have shipped:

| Bug | What it looked like | What it was |
|---|---|---|
| Session-based caching | Ran fine, control agent at 0% hit rate | Wrong model of how caching works |
| Prompt bloat invisible | Ran fine, flat line where growth was seeded | Detector measuring the wrong column |
| Overlap double-count | Ran fine, produced a *bigger, better* number | Claiming >100% of one agent's spend |

The third is the instructive one, because the bug made the product look
**better**. A larger headline number is exactly what you want to see, so there
is no instinct to check it. It was caught only by printing two evidence tables
next to each other and dividing one column by another.

**The habit that generalises:** after generating code, do not ask "did it run?"
Ask "what would this look like if it were wrong?" — then go and measure that
specific thing. All three bugs were invisible to the interpreter and obvious to
a five-line probe.
