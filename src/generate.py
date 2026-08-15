"""
Synthetic usage-log generator.

Purpose: give the app a believable file to audit when you cannot show a real
bill on camera, and give the detectors something with a KNOWN right answer to
be tested against.

Two rules kept this honest:

  1. The generator never writes a "this row is wasteful" flag. It simulates
     plausible engineering behaviour and lets waste EMERGE. If a detector finds
     the cache-miss pattern, it is because the pattern is really in the token
     counts, not because it was labelled.

  2. Not everything is broken. `invoice_parser` caches properly and runs on a
     budget model. A report where every single agent is guilty reads as fake,
     and the contrast is what makes the guilty ones legible.

The scenario: "Northwind", a mid-size company six weeks into shipping LLM
features, with the cost mistakes a team makes in its first quarter.

Usage:
    python src/generate.py --weeks 6 --seed 7 --format canonical -o data/usage.csv
    python src/generate.py --format anthropic -o data/raw_anthropic.csv
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Agent behaviour profiles
# ---------------------------------------------------------------------------

@dataclass
class AgentProfile:
    name: str
    model: str
    provider: str
    calls_per_weekday: int
    weekend_factor: float
    hours: tuple[int, ...]              # local hours this agent is active
    prefix_tokens: int                  # stable system prompt + tool defs
    n_prefix_variants: int              # distinct system prompts in rotation
    user_tokens: tuple[int, int]        # lognormal-ish (median, spread)
    output_tokens: tuple[int, int]
    uses_cache: bool
    is_batch: bool = False
    error_rate: float = 0.01
    truncation_rate: float = 0.0
    session_len: tuple[int, int] = (1, 1)   # calls per session (min, max)
    loop_prob: float = 0.0                  # chance a session degenerates
    prefix_growth_per_week: float = 0.0     # prompt bloat, fraction per week
    session_id_missing: float = 0.0
    notes: str = ""


PROFILES: list[AgentProfile] = [
    # --- guilty: frontier model doing trivial classification -----------------
    AgentProfile(
        name="support_triage",
        model="claude-opus-5", provider="anthropic",
        calls_per_weekday=420, weekend_factor=0.35,
        hours=tuple(range(7, 21)),
        prefix_tokens=900, n_prefix_variants=1,
        user_tokens=(320, 140), output_tokens=(28, 12),
        uses_cache=False,
        error_rate=0.008,
        notes="Routes tickets into 9 buckets. Outputs a single label on a "
              "frontier model — the classic over-provisioning shape.",
    ),
    # --- guilty: enormous stable prefix, caching never switched on ----------
    AgentProfile(
        name="doc_summariser",
        model="claude-sonnet-4-5", provider="anthropic",
        calls_per_weekday=260, weekend_factor=0.2,
        hours=tuple(range(8, 19)),
        prefix_tokens=6400, n_prefix_variants=2,
        user_tokens=(1500, 700), output_tokens=(420, 180),
        uses_cache=False,
        error_rate=0.012, truncation_rate=0.06,
        notes="6.4k tokens of style guide + few-shot examples re-sent on every "
              "call. Biggest single recoverable line item.",
    ),
    # --- guilty: agentic loops that sometimes never terminate ---------------
    AgentProfile(
        name="research_v2",
        model="claude-opus-5", provider="anthropic",
        calls_per_weekday=55, weekend_factor=0.5,
        hours=tuple(range(9, 23)),
        prefix_tokens=3100, n_prefix_variants=1,
        user_tokens=(2200, 1100), output_tokens=(300, 200),
        uses_cache=True,
        error_rate=0.03, truncation_rate=0.04,
        session_len=(4, 14), loop_prob=0.05,
        notes="Multi-step agent. ~5% of runs hit a tool that returns the same "
              "result forever and the agent keeps paying to re-read it.",
    ),
    # --- guilty: latency-tolerant nightly job on the sync endpoint ----------
    AgentProfile(
        name="nightly_enrichment",
        model="gpt-5", provider="openai",
        calls_per_weekday=900, weekend_factor=1.0,
        hours=(1, 2, 3, 4),
        prefix_tokens=700, n_prefix_variants=1,
        user_tokens=(600, 240), output_tokens=(150, 60),
        uses_cache=False, is_batch=False,
        error_rate=0.02,
        session_id_missing=0.9,
        notes="Backfills metadata at 1am. Nobody is waiting for it, yet it "
              "runs on the real-time endpoint at full price.",
    ),
    # --- guilty: system prompt that grew every sprint -----------------------
    AgentProfile(
        name="chat_api",
        model="gpt-5", provider="openai",
        calls_per_weekday=1350, weekend_factor=0.55,
        hours=tuple(range(6, 24)),
        prefix_tokens=1200, n_prefix_variants=1,
        user_tokens=(240, 160), output_tokens=(210, 130),
        uses_cache=True,
        error_rate=0.015, truncation_rate=0.02,
        session_len=(2, 6),
        prefix_growth_per_week=0.19,
        notes="Product surface. Every sprint someone appended a rule to the "
              "system prompt; nobody ever removed one.",
    ),
    # --- the control: a team that did it right ------------------------------
    AgentProfile(
        name="invoice_parser",
        model="claude-haiku-4-5", provider="anthropic",
        calls_per_weekday=700, weekend_factor=0.15,
        hours=tuple(range(8, 20)),
        prefix_tokens=2800, n_prefix_variants=1,
        user_tokens=(900, 380), output_tokens=(120, 40),
        uses_cache=True, is_batch=True,
        error_rate=0.006,
        notes="Cached prefix, budget model, batch endpoint. Should surface as "
              "clean — the contrast that makes the other findings readable.",
    ),
]


# ---------------------------------------------------------------------------

def _lognormal(rng: np.random.Generator, median: float, spread: float) -> int:
    """Token counts are right-skewed: many small, a few very large."""
    sigma = max(0.15, min(1.2, spread / max(median, 1.0)))
    val = rng.lognormal(mean=math.log(max(median, 1.0)), sigma=sigma)
    return int(max(1, round(val)))


def _hash(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _diurnal_weight(hour: int) -> float:
    """Gentle mid-morning and mid-afternoon humps."""
    return 1.0 + 0.35 * math.sin((hour - 6) / 24 * 2 * math.pi) + 0.15 * math.cos(hour / 6)


ERROR_CODES = [
    ("rate_limit_error", 0.42), ("overloaded_error", 0.24),
    ("api_error", 0.16), ("invalid_request_error", 0.10),
    ("timeout", 0.08),
]


def _sample_error(rng: np.random.Generator) -> str:
    codes, weights = zip(*ERROR_CODES)
    return str(rng.choice(codes, p=np.array(weights) / sum(weights)))


CACHE_TTL_SECONDS = 300  # 5-minute default TTL


def _apply_cache_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """Assign cache reads/writes the way prompt caching actually behaves.

    An earlier version tied cache hits to position within a session (first call
    writes, rest read). That was wrong twice over:

      * it gave zero cache hits to any high-volume agent whose sessions are a
        single call each, even though such an agent benefits from caching most;
      * a cache entry lives on a TTL, not a session, so a hit depends on wall
        time since the prefix was last touched — across sessions.

    So: walk the merged timeline once, keyed on (agent, model, prefix). If the
    prefix was touched within the TTL it is a read; otherwise it is a write.
    """
    df = df.sort_values("ts").reset_index(drop=True)

    last_seen: dict[tuple, pd.Timestamp] = {}
    reads = np.zeros(len(df), dtype=np.int64)
    writes = np.zeros(len(df), dtype=np.int64)
    ttls: list[str] = []
    billed_input = np.zeros(len(df), dtype=np.int64)

    ts = df["ts"].to_numpy()
    agents = df["agent"].to_numpy()
    models = df["model"].to_numpy()
    prefixes = df["prompt_prefix_hash"].to_numpy()
    prefix_toks = df["_prefix_tokens"].to_numpy()
    user_toks = df["_user_tokens"].to_numpy()
    uses_cache = df["_uses_cache"].to_numpy()

    for i in range(len(df)):
        if not uses_cache[i]:
            # Caching off: the whole prefix is re-billed at full rate, forever.
            billed_input[i] = prefix_toks[i] + user_toks[i]
            ttls.append("")
            continue

        key = (agents[i], models[i], prefixes[i])
        prev = last_seen.get(key)
        now = ts[i]
        warm = prev is not None and (now - prev) / np.timedelta64(1, "s") <= CACHE_TTL_SECONDS

        if warm:
            reads[i] = prefix_toks[i]
            ttls.append("")
        else:
            writes[i] = prefix_toks[i]
            ttls.append("5m")
        billed_input[i] = user_toks[i]
        last_seen[key] = now

    df["input_tokens"] = billed_input
    df["cache_read_tokens"] = reads
    df["cache_write_tokens"] = writes
    df["cache_write_ttl"] = ttls
    df["prompt_chars"] = (
        (df["input_tokens"] + df["cache_read_tokens"] + df["cache_write_tokens"]) * 3.9
    ).astype(int)

    return df.drop(columns=["_prefix_tokens", "_user_tokens", "_uses_cache"])


def generate(
    weeks: int = 6,
    seed: int = 7,
    end: Optional[datetime] = None,
    include_unknown_model: bool = True,
    scale: float = 1.0,
) -> pd.DataFrame:
    """`scale` multiplies call volume. The default models an early-stage team
    (~150k calls / 6 weeks). Raise it to model a larger org; row count and
    runtime grow linearly."""
    rng = np.random.default_rng(seed)
    end = end or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(weeks=weeks)

    rows: list[dict] = []
    req_n = 0

    for prof in PROFILES:
        # Stable prefix identities for this agent, fixed for the whole period.
        prefix_ids = [_hash(prof.name, "prefix", i) for i in range(prof.n_prefix_variants)]

        day = start
        while day < end:
            week_idx = (day - start).days // 7
            is_weekend = day.weekday() >= 5
            base = prof.calls_per_weekday * (prof.weekend_factor if is_weekend else 1.0) * scale
            # Organisations grow; volume drifts up slowly with weekly noise.
            growth = 1.0 + 0.045 * week_idx
            n_calls = max(0, int(rng.normal(base * growth, base * 0.16)))

            # Prompt bloat: the prefix itself gets bigger week over week.
            prefix_tokens = int(
                prof.prefix_tokens * (1.0 + prof.prefix_growth_per_week) ** week_idx
            )

            # Group calls into sessions.
            remaining = n_calls
            while remaining > 0:
                s_len = int(rng.integers(prof.session_len[0], prof.session_len[1] + 1))
                s_len = min(s_len, remaining)
                remaining -= s_len

                session_id = _hash(prof.name, day.date(), rng.integers(0, 1 << 30))
                if rng.random() < prof.session_id_missing:
                    session_id = None

                # A runaway session: the agent re-sends an identical prompt
                # dozens of times. Nothing labels this — it is just a long run
                # of repeated prompt_full_hash within one session_id.
                runaway = rng.random() < prof.loop_prob
                if runaway:
                    s_len = int(rng.integers(45, 140))
                    stuck_hash = _hash(prof.name, session_id, "stuck")
                    stuck_user = _lognormal(rng, prof.user_tokens[0], prof.user_tokens[1])

                hour = int(rng.choice(
                    prof.hours,
                    p=np.array([_diurnal_weight(h) for h in prof.hours])
                      / sum(_diurnal_weight(h) for h in prof.hours),
                ))
                t = day.replace(hour=hour) + timedelta(
                    minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60))
                )

                prefix_id = prefix_ids[int(rng.integers(0, len(prefix_ids)))]
                model = prof.model
                # Mid-period, part of chat_api's traffic migrates to a cheaper
                # model — so the log is not perfectly homogeneous per agent.
                if prof.name == "chat_api" and week_idx >= 3 and rng.random() < 0.22:
                    model = "gpt-5-mini"
                if include_unknown_model and rng.random() < 0.0015:
                    model = "gpt-4o-legacy"  # deliberately absent from rate_cards

                for i in range(s_len):
                    req_n += 1
                    t = t + timedelta(seconds=int(rng.integers(2, 90)))

                    if runaway:
                        user_toks = stuck_user
                        full_hash = stuck_hash
                        out_toks = _lognormal(rng, 60, 25)
                    else:
                        user_toks = _lognormal(rng, prof.user_tokens[0], prof.user_tokens[1])
                        full_hash = _hash(prof.name, session_id, i, rng.integers(0, 1 << 30))
                        out_toks = _lognormal(rng, prof.output_tokens[0], prof.output_tokens[1])

                    # Cache accounting is deferred to _apply_cache_semantics()
                    # below, because a hit depends on whether this prefix was
                    # touched within the TTL window — which is only knowable
                    # once every agent's rows are merged into one timeline.
                    status, err, finish = "ok", "", "stop"
                    if rng.random() < prof.error_rate:
                        status, err, finish = "error", _sample_error(rng), "error"
                        # Errors still bill input on most providers; output is lost.
                        out_toks = 0
                    elif rng.random() < prof.truncation_rate:
                        finish = "length"
                        out_toks = int(out_toks * rng.uniform(1.8, 2.6))

                    rows.append({
                        "ts": t,
                        "request_id": f"req-{req_n:08d}",
                        "session_id": session_id,
                        "agent": prof.name,
                        "provider": prof.provider,
                        "model": model,
                        "output_tokens": out_toks,
                        "status": status,
                        "error_code": err,
                        "finish_reason": finish,
                        "latency_ms": int(max(120, rng.normal(
                            900 + out_toks * 7, 260))),
                        "is_batch": prof.is_batch,
                        "prompt_prefix_hash": prefix_id,
                        "prompt_full_hash": full_hash,
                        # staging fields, consumed by _apply_cache_semantics
                        "_prefix_tokens": prefix_tokens,
                        "_user_tokens": user_toks,
                        "_uses_cache": prof.uses_cache,
                    })

            day += timedelta(days=1)

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df = _apply_cache_semantics(df)

    # Real exports have gaps. Punch a few holes so the app's data-quality
    # panel has something true to report.
    if len(df):
        holes = rng.random(len(df)) < 0.004
        df.loc[holes, "prompt_full_hash"] = None
        holes2 = rng.random(len(df)) < 0.002
        df.loc[holes2, "latency_ms"] = None

    from schema import ALL_COLUMNS
    return df[list(ALL_COLUMNS)]


# ---------------------------------------------------------------------------
# Emit in provider-native shapes, so normalize.py gets exercised for real
# ---------------------------------------------------------------------------

def to_anthropic_export(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["provider"] == "anthropic"].copy()
    return pd.DataFrame({
        "start_time": d["ts"],
        "request_id": d["request_id"],
        "workspace": d["agent"],
        "model": d["model"],
        "input_tokens": d["input_tokens"],                       # excludes cache reads
        "output_tokens": d["output_tokens"],
        "cache_read_input_tokens": d["cache_read_tokens"],
        "cache_creation_input_tokens": d["cache_write_tokens"],
        "stop_reason": d["finish_reason"],
        "status": d["status"],
        "session_id": d["session_id"],
        "prompt_prefix_hash": d["prompt_prefix_hash"],
        "prompt_full_hash": d["prompt_full_hash"],
    })


def to_openai_export(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["provider"] == "openai"].copy()
    return pd.DataFrame({
        "start_time": d["ts"],
        "request_id": d["request_id"],
        "project_name": d["agent"],
        "model": d["model"],
        # OpenAI convention: input_tokens INCLUDES cached_tokens.
        "input_tokens": d["input_tokens"] + d["cache_read_tokens"],
        "cached_tokens": d["cache_read_tokens"],
        "output_tokens": d["output_tokens"],
        "finish_reason": d["finish_reason"],
        "status": d["status"],
        "session_id": d["session_id"],
        "prompt_prefix_hash": d["prompt_prefix_hash"],
        "prompt_full_hash": d["prompt_full_hash"],
    })


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic LLM usage log.")
    p.add_argument("--weeks", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiply call volume (1.0 = early-stage team).")
    p.add_argument("--format", choices=["canonical", "anthropic", "openai"],
                   default="canonical")
    p.add_argument("-o", "--out", default="data/usage_canonical.csv")
    args = p.parse_args()

    df = generate(weeks=args.weeks, seed=args.seed, scale=args.scale)
    out = {
        "canonical": lambda: df,
        "anthropic": lambda: to_anthropic_export(df),
        "openai": lambda: to_openai_export(df),
    }[args.format]()

    out.to_csv(args.out, index=False)
    print(f"wrote {len(out):,} rows -> {args.out}  (format={args.format})")


if __name__ == "__main__":
    main()
