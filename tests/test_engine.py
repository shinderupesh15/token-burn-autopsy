"""
Day 2 tests: pricing, the six detectors, and the de-overlap arithmetic.

The tests that matter most are the invariants at the bottom. A cost report is
only worth anything if it cannot claim more money than exists, and that is a
property you assert, not a thing you eyeball once.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine import run_audit                                    # noqa: E402
from generate import generate                                   # noqa: E402
from rates import load_rate_card, price, to_monthly, unpriced_report  # noqa: E402
from schema import conform                                      # noqa: E402
from summary import action_list, exec_summary, verdict          # noqa: E402


@pytest.fixture(scope="module")
def rc():
    return load_rate_card(ROOT / "rate_cards.yaml")


@pytest.fixture(scope="module")
def priced(rc):
    return price(conform(generate(weeks=6, seed=7)), rc)


@pytest.fixture(scope="module")
def audit(priced, rc):
    return run_audit(priced, rc)


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

def test_cache_reads_priced_at_documented_multiplier(rc):
    """Anthropic cache reads bill at 0.1x input. The whole cache finding rests
    on this number, so it is asserted against the rate card directly."""
    assert rc.multipliers("anthropic")["cache_read"] == 0.1
    assert rc.multipliers("anthropic")["cache_write_5m"] == 1.25
    assert rc.multipliers("anthropic")["cache_write_1h"] == 2.0
    assert rc.multipliers("anthropic")["batch"] == 0.5
    assert rc.multipliers("openai")["batch"] == 0.5


def test_provider_cache_discounts_are_not_assumed_equal(rc):
    """OpenAI's cached-input discount is much weaker than Anthropic's.
    Assuming one number across providers would overstate every OpenAI saving."""
    assert rc.multipliers("openai")["cache_read"] != rc.multipliers("anthropic")["cache_read"]


def test_unknown_provider_falls_back_conservatively(rc):
    """An unrecognised provider must not invent a discount."""
    m = rc.multipliers("some-new-vendor")
    assert m["cache_read"] == 1.0 and m["batch"] == 1.0


def test_unpriced_models_excluded_not_zeroed(priced):
    """A model missing from the rate card must be reported, not silently
    contribute $0 to a total that then reads as complete."""
    bad = priced[~priced["priced"]]
    assert len(bad) > 0, "fixture should contain an unpriced model"
    assert (bad["cost_total"] == 0).all()
    rep = unpriced_report(priced)
    assert "gpt-4o-legacy" in set(rep["model"])
    assert rep["calls"].sum() == len(bad)


def test_batch_rows_are_discounted(priced):
    batched = priced[priced.is_batch.fillna(False)]
    assert len(batched) > 0
    assert (batched["batch_mult"] == 0.5).all()


def test_cost_components_sum_to_total(priced):
    parts = (priced.cost_input + priced.cost_output
             + priced.cost_cache_read + priced.cost_cache_write)
    assert (parts - priced.cost_total).abs().max() < 1e-9


def test_monthly_normalisation():
    assert to_monthly(100.0, 30.0) == pytest.approx(100.0)
    assert to_monthly(100.0, 60.0) == pytest.approx(50.0)


# --------------------------------------------------------------------------
# Detectors fire on the seeded patterns
# --------------------------------------------------------------------------

def test_all_six_detectors_fire(audit):
    keys = {f.key for f in audit.findings}
    assert keys == {"cache_miss", "right_sizing", "dead_spend",
                    "runaway_loop", "batchable", "prompt_bloat"}


def test_cache_detector_picks_the_cheaper_ttl(audit):
    """Sparse traffic misses a 5-minute window every time. The detector has to
    evaluate the 1-hour TTL too, or it under-reports the biggest finding."""
    f = next(f for f in audit.findings if f.key == "cache_miss")
    assert "best_ttl" in f.evidence.columns
    assert (f.evidence["best_ttl"] == "1h").any()
    # doc_summariser's 7k-token prefix must be found
    assert f.evidence["est_prefix_tokens"].max() > 6_000


def test_cache_detector_ignores_small_prefixes(audit, rc):
    """nightly_enrichment has a 700-token prefix — below the threshold where
    caching pays for its write premium. It must not be flagged here."""
    f = next(f for f in audit.findings if f.key == "cache_miss")
    floor = rc.thresholds["cache_miss"]["min_prefix_tokens"]
    assert (f.evidence["est_prefix_tokens"] >= floor).all()
    assert "nightly_enrichment" not in set(f.evidence["agent"])


def test_right_sizing_targets_the_classifier(audit):
    f = next(f for f in audit.findings if f.key == "right_sizing")
    row = f.evidence[f.evidence.agent == "support_triage"].iloc[0]
    assert row["model"] == "claude-opus-5"
    assert row["suggested"] == "claude-sonnet-4-5"
    assert row["median_output_tokens"] < 60
    assert row["share_of_agent"] > 0.9      # essentially all of its traffic


def test_right_sizing_is_marked_estimated(audit):
    """This is the one finding that can cost quality. It must never be
    presented with the same confidence as a published discount."""
    f = next(f for f in audit.findings if f.key == "right_sizing")
    assert f.confidence == "estimated"
    assert "eval" in f.fix.lower()


def test_batchable_only_flags_off_hours_agents(audit):
    f = next(f for f in audit.findings if f.key == "batchable")
    assert set(f.evidence["agent"]) == {"nightly_enrichment"}
    assert (f.evidence["off_hours_share"] >= 0.9).all()


def test_batchable_excludes_already_batched_agents(audit):
    f = next(f for f in audit.findings if f.key == "batchable")
    assert "invoice_parser" not in set(f.evidence["agent"])


def test_runaway_loop_is_always_critical(audit):
    """Small in dollars, large in meaning — severity is pinned regardless of
    its share of spend."""
    f = next(f for f in audit.findings if f.key == "runaway_loop")
    assert f.severity == "critical"
    assert f.evidence["identical_calls"].max() >= 40


def test_dead_spend_covers_errors_and_truncations(audit):
    f = next(f for f in audit.findings if f.key == "dead_spend")
    reasons = set(f.evidence["reason"])
    assert any("truncated" in r for r in reasons)
    assert any("rate_limit" in r for r in reasons)


def test_prompt_bloat_finds_chat_api(audit):
    f = next(f for f in audit.findings if f.key == "prompt_bloat")
    row = f.evidence[f.evidence.agent == "chat_api"].iloc[0]
    assert row["growth"] > 0.8
    assert row["latest_tokens"] > row["week_1_tokens"]


def test_clean_agent_is_barely_flagged(audit, priced):
    """The control must survive the whole engine nearly untouched, or the
    report has no contrast and reads as fabricated."""
    per_agent = audit.by_agent(priced).set_index("agent")["recoverable"]
    spend = priced.groupby("agent")["cost_total"].sum()
    assert per_agent.get("invoice_parser", 0.0) / spend["invoice_parser"] < 0.02


# --------------------------------------------------------------------------
# The invariants — a report may never claim money that does not exist
# --------------------------------------------------------------------------

def test_overlap_is_corrected_not_summed(audit):
    """nightly_enrichment is flagged as BOTH over-provisioned and batchable.
    Summing those claims more than that agent ever cost. The de-overlapped
    total must come in strictly below the naive sum."""
    assert audit.naive_recoverable > audit.recoverable
    assert audit.overlap_correction > 0


def test_recoverable_never_exceeds_spend(audit):
    assert audit.recoverable <= audit.total_spend + 1e-9
    assert 0.0 <= audit.recoverable_share <= 1.0


def test_no_agent_is_claimed_beyond_its_own_spend(audit, priced):
    """The bug this guards: a naive sum let one nightly job be 'saved' twice,
    producing more recoverable dollars than the job cost."""
    waste = audit.by_agent(priced).set_index("agent")["recoverable"]
    spend = priced.groupby("agent")["cost_total"].sum()
    for agent, w in waste.items():
        assert w <= spend[agent] + 1e-9, f"{agent}: claimed {w} of {spend[agent]}"


def test_no_single_request_is_saved_twice(audit, priced):
    cost = priced.set_index("request_id")["cost_total"]
    combined = audit.row_combined
    assert (combined <= cost.reindex(combined.index) + 1e-9).all()


def test_leaderboard_sums_to_the_headline(audit, priced):
    """If the bars do not add up to the hero number, the chart is lying."""
    assert audit.by_agent(priced)["recoverable"].sum() == pytest.approx(
        audit.recoverable, rel=1e-6)


def test_findings_sorted_by_size(audit):
    vals = [f.recoverable for f in audit.findings]
    assert vals == sorted(vals, reverse=True)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def test_summary_escapes_dollar_signs(audit):
    """Streamlit renders markdown with LaTeX on: an unescaped pair of dollar
    signs on one line silently became an equation."""
    text = exec_summary(audit)
    assert "$" in text
    assert text.count("\\$") == text.count("$")


def test_summary_states_the_overlap_correction(audit):
    text = exec_summary(audit)
    assert "overlap" in text.lower()


def test_summary_flags_estimated_findings(audit):
    assert "validate before acting" in exec_summary(audit).lower()


def test_verdict_scales_with_severity(audit):
    assert verdict(audit)[0] == "Substantial waste"   # fixture is ~54%


def test_action_list_is_ranked_and_complete(audit):
    acts = action_list(audit)
    assert len(acts) == len(audit.findings)
    assert [a["rank"] for a in acts] == list(range(1, len(acts) + 1))
    assert all(a["action"].endswith(".") for a in acts)


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------

def test_clean_log_produces_few_findings(rc):
    """A log containing only the well-behaved agent should not be talked into
    a long list of problems."""
    df = generate(weeks=6, seed=11)
    clean = df[df.agent == "invoice_parser"]
    audit = run_audit(price(conform(clean), rc), rc)
    assert audit.recoverable_share < 0.05


def test_tiny_log_does_not_crash(rc):
    df = generate(weeks=6, seed=5).head(20)
    audit = run_audit(price(conform(df), rc), rc)
    assert audit.recoverable >= 0
    assert isinstance(exec_summary(audit), str)
