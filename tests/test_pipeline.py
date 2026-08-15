"""
Pipeline tests: the generated log must actually contain the six waste patterns,
and every adapter must round-trip without corrupting token counts.

This suite is the reason the savings engine can be trusted downstream. If the
fixture does not provably contain a pattern, a detector "finding" it proves
nothing.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate import generate, to_anthropic_export, to_openai_export  # noqa: E402
from normalize import normalize  # noqa: E402
from schema import ALL_COLUMNS, REQUIRED, SchemaError, conform, validate  # noqa: E402


@pytest.fixture(scope="module")
def log() -> pd.DataFrame:
    return generate(weeks=6, seed=7)


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def test_generated_log_matches_schema(log):
    assert list(log.columns) == list(ALL_COLUMNS)
    assert len(log) > 50_000


def test_conform_rejects_missing_required_column(log):
    with pytest.raises(SchemaError, match="missing required column"):
        conform(log.drop(columns=["input_tokens"]), source="test")


def test_no_negative_token_counts(log):
    cols = ["input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"]
    assert (log[cols] >= 0).all().all()


def test_validate_reports_short_window():
    short = generate(weeks=1, seed=3)
    assert any("spans" in w for w in validate(conform(short)))


# --------------------------------------------------------------------------
# Adapters — the token-convention trap
# --------------------------------------------------------------------------

def test_openai_adapter_subtracts_cached_from_input(log):
    """OpenAI reports cached tokens INSIDE input_tokens. If the adapter forgets
    to subtract, every cache saving estimate is inflated."""
    raw = to_openai_export(log)
    got = normalize(raw)
    assert got.adapter == "openai"

    merged = got.df.set_index("request_id")["input_tokens"].astype("int64")
    expected = log.set_index("request_id")["input_tokens"].reindex(merged.index)
    assert (merged == expected).all()

    # And the raw file really did double-count, or this test proves nothing.
    raw_in = raw.set_index("request_id")["input_tokens"]
    cached = raw.set_index("request_id")["cached_tokens"]
    assert (cached > 0).any()
    assert (raw_in >= cached).all()


def test_anthropic_adapter_preserves_input(log):
    got = normalize(to_anthropic_export(log))
    assert got.adapter == "anthropic"
    merged = got.df.set_index("request_id")["input_tokens"].astype("int64")
    expected = log.set_index("request_id")["input_tokens"].reindex(merged.index)
    assert (merged == expected).all()


def test_unrecognised_file_raises():
    with pytest.raises(ValueError, match="Could not identify"):
        normalize(pd.DataFrame({"foo": [1], "bar": [2]}))


# --------------------------------------------------------------------------
# The six patterns must be present and measurable
# --------------------------------------------------------------------------

def test_pattern_1_cache_miss_waste(log):
    """Three agents re-send a large stable prefix with caching off."""
    per_agent = log.groupby("agent")[["cache_read_tokens", "cache_write_tokens"]].sum().sum(axis=1)
    uncached = set(per_agent[per_agent == 0].index)
    assert {"doc_summariser", "support_triage", "nightly_enrichment"} <= uncached

    doc = log[log.agent == "doc_summariser"]
    assert doc["input_tokens"].mean() > 5_000  # prefix is riding in every call


def test_pattern_2_model_over_provisioning(log):
    """support_triage emits ~30 tokens per call on a frontier model."""
    st = log[log.agent == "support_triage"]
    assert st["model"].mode()[0] == "claude-opus-5"
    assert st["output_tokens"].mean() < 60
    assert len(st) > 10_000


def test_pattern_3_dead_spend(log):
    assert (log.status == "error").sum() > 500
    assert (log.finish_reason == "length").sum() > 500
    # Errors bill input but produce nothing.
    errs = log[log.status == "error"]
    assert (errs["output_tokens"] == 0).all()
    assert errs["input_tokens"].sum() > 0


def test_pattern_4_runaway_loops(log):
    sized = (
        log.dropna(subset=["session_id", "prompt_full_hash"])
        .groupby(["session_id", "prompt_full_hash"])
        .size()
    )
    assert (sized >= 10).sum() >= 5
    assert sized.max() >= 40  # at least one badly stuck run


def test_pattern_5_batchable_workload(log):
    ne = log[log.agent == "nightly_enrichment"]
    assert len(ne) > 20_000
    assert not ne["is_batch"].any()              # paying full price
    assert ne["ts"].dt.hour.between(1, 6).mean() > 0.95  # nobody is waiting


def test_pattern_6_prompt_bloat_drift(log):
    """chat_api's prefix grows ~19%/week. Because that agent caches, the growth
    shows up in TOTAL prompt tokens, not in input_tokens — the detector has to
    measure the whole input side or it will miss this entirely."""
    ch = log[log.agent == "chat_api"].copy()
    ch["total_prompt"] = ch.input_tokens + ch.cache_read_tokens + ch.cache_write_tokens
    ch["week"] = (ch.ts - ch.ts.min()).dt.days // 7
    weekly = ch.groupby("week")["total_prompt"].mean()

    assert weekly.is_monotonic_increasing
    assert weekly.iloc[-1] / weekly.iloc[0] > 1.8

    # Guard against a false-positive detector: input_tokens alone stays flat.
    flat = ch.groupby("week")["input_tokens"].mean()
    assert abs(flat.iloc[-1] / flat.iloc[0] - 1) < 0.25


def test_control_agent_is_clean(log):
    """A report that indicts every agent is not credible. invoice_parser must
    come out innocent on all six counts."""
    inv = log[log.agent == "invoice_parser"]
    assert (inv["cache_read_tokens"] > 0).mean() > 0.9   # caching works
    assert inv["is_batch"].all()                          # batched
    assert inv["model"].mode()[0] == "claude-haiku-4-5"   # right-sized
    assert (inv.status == "error").mean() < 0.02


# --------------------------------------------------------------------------
# Cache semantics
# --------------------------------------------------------------------------

def test_cache_hits_follow_ttl_not_session(log):
    """Regression test. The first version tied cache hits to position within a
    session, which gave a 0% hit rate to single-call agents that should cache
    best. Hits must depend on wall-clock recency of the prefix instead."""
    inv = log[log.agent == "invoice_parser"].sort_values("ts")
    singles = inv.groupby("session_id").size()
    assert (singles == 1).mean() > 0.9          # sessions really are single calls
    assert (inv["cache_read_tokens"] > 0).mean() > 0.9   # yet it still hits cache


def test_reads_and_writes_are_mutually_exclusive(log):
    both = (log.cache_read_tokens > 0) & (log.cache_write_tokens > 0)
    assert not both.any()


def test_determinism():
    a = generate(weeks=2, seed=42, end=pd.Timestamp("2026-08-01", tz="UTC").to_pydatetime())
    b = generate(weeks=2, seed=42, end=pd.Timestamp("2026-08-01", tz="UTC").to_pydatetime())
    pd.testing.assert_frame_equal(a, b)
