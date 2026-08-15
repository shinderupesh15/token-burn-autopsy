"""
The savings engine: six detectors over a priced canonical log.

Every Finding carries a `basis` string stating the arithmetic in words, and a
`confidence` that is honest about which numbers are provider-documented and
which are projections:

    documented  the lever's discount is published by the provider; the only
                estimate is how many tokens qualify
    estimated   the recommendation needs human validation before you act
                (e.g. a model swap requires an eval to confirm quality holds)

Reporting an estimated saving with the same authority as a documented one is
how a cost audit loses credibility the first time someone checks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from rates import RateCard, period_days, to_monthly


@dataclass
class Finding:
    key: str
    title: str
    headline: str                     # one line, plain language
    recoverable: float                # standalone, over the log period
    monthly: float                    # normalised to 30 days
    calls_affected: int
    agents: list[str]
    fix: str
    basis: str                        # the arithmetic, in words
    confidence: str                   # documented | estimated
    evidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    severity: str = "medium"          # critical | high | medium | low
    detail: dict[str, Any] = field(default_factory=dict)
    # Per-request saving, indexed by request_id. Required for de-overlapping:
    # two findings can name the same call, and adding their savings together
    # can claim more than that call ever cost.
    row_savings: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _spread(g: pd.DataFrame, total: float) -> pd.Series:
    """Distribute a group-level saving across its rows, in proportion to cost."""
    cost = g["cost_total"].astype(float)
    denom = float(cost.sum())
    weights = cost / denom if denom > 0 else pd.Series(1.0 / len(g), index=g.index)
    return pd.Series((weights * total).to_numpy(), index=g["request_id"].to_numpy())


def _severity(monthly: float, total_monthly: float) -> str:
    if total_monthly <= 0:
        return "low"
    share = monthly / total_monthly
    if share >= 0.20:
        return "critical"
    if share >= 0.10:
        return "high"
    if share >= 0.03:
        return "medium"
    return "low"


def _fmt(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 100 else f"${x:,.2f}"


# ---------------------------------------------------------------------------
# 1. Cache-miss waste
# ---------------------------------------------------------------------------

def detect_cache_miss(df: pd.DataFrame, rc: RateCard, days: float) -> Optional[Finding]:
    th = rc.thresholds.get("cache_miss", {})
    min_repeats = int(th.get("min_repeats", 5))

    # Only rows that never touched the cache can be missing cache savings.
    cand = df[(df.cache_read_tokens.fillna(0) == 0)
              & (df.cache_write_tokens.fillna(0) == 0)
              & df.priced].copy()
    if cand.empty:
        return None

    group_cols = ["agent", "provider", "model", "prompt_prefix_hash"]
    if cand["prompt_prefix_hash"].isna().all():
        group_cols = ["agent", "provider", "model"]
        cand["prompt_prefix_hash"] = "(unknown)"

    rows: list[dict] = []
    savings_parts: list[pd.Series] = []
    for keys, g in cand.groupby(group_cols, dropna=False):
        if len(g) < min_repeats:
            continue
        # The stable prefix is the floor of the input distribution: the
        # smallest call in the group is essentially prefix + a tiny user turn.
        # The 5th percentile is used instead of the true min so one anomalous
        # short call cannot collapse the estimate.
        prefix_tokens = float(np.percentile(g["input_tokens"].astype(float), 5))
        provider, model = str(keys[1]), str(keys[2])
        min_prefix = rc.cache_minimum_tokens(provider, model)
        if prefix_tokens < min_prefix:
            continue

        rate = float(g["base_input_rate"].iloc[0])
        mult = rc.multipliers(str(g["provider"].iloc[0]))
        n = len(g)
        current = n * prefix_tokens / 1e6 * rate

        # A cache entry survives its TTL, so the writes you actually pay for is
        # the number of distinct TTL windows containing traffic, not the number
        # of calls. Which TTL wins depends on how the agent's traffic is spaced:
        # a busy agent is fine on the cheap 5-minute write, but an agent whose
        # calls are minutes apart misses that window every time and does better
        # paying the 2x hourly write once. Both are evaluated; cheaper wins.
        best = None
        for label, freq, wmult in (("5m", "5min", mult["cache_write_5m"]),
                                   ("1h", "1h", mult["cache_write_1h"])):
            windows = g["ts"].dt.floor(freq).nunique()
            reads = max(n - windows, 0)
            cost = (windows * prefix_tokens / 1e6 * rate * wmult
                    + reads * prefix_tokens / 1e6 * rate * mult["cache_read"])
            if best is None or cost < best[2]:
                best = (label, windows, cost)

        ttl_label, windows, cached = best
        saving = current - cached
        if saving <= 0:
            continue

        rows.append({
            "agent": keys[0], "provider": provider, "model": model,
            "calls": n, "est_prefix_tokens": int(prefix_tokens),
            "min_cache_tokens": min_prefix,
            "best_ttl": ttl_label, "cache_windows": int(windows),
            "paid": current, "if_cached": cached, "recoverable": saving,
        })
        savings_parts.append(_spread(g, saving))

    if not rows:
        return None

    ev = pd.DataFrame(rows).sort_values("recoverable", ascending=False)
    total = float(ev["recoverable"].sum())
    agents = ev["agent"].drop_duplicates().tolist()
    worst = ev.iloc[0]

    return Finding(
        key="cache_miss",
        title="Uncached prompt prefixes",
        headline=(
            f"{len(agents)} agent(s) re-send a large stable prefix on every call. "
            f"Worst: {worst['agent']} repeats ~{int(worst['est_prefix_tokens']):,} "
            f"tokens across {int(worst['calls']):,} calls."
        ),
        recoverable=total,
        monthly=to_monthly(total, days),
        calls_affected=int(ev["calls"].sum()),
        agents=agents,
        fix=("Mark the system prompt and tool definitions as a cache breakpoint. "
             "No prompt rewriting needed — the content is already identical. "
             "Check the suggested TTL per row: sparse traffic needs the 1-hour "
             "window to hit at all."),
        basis=("Prefix size estimated as the 5th percentile of input tokens per "
               "(agent, model, prefix). Re-priced as one cache write per active "
               "TTL window plus remaining calls at the cache-read multiplier; "
               "5-minute and 1-hour TTLs both evaluated, cheaper reported. "
               "Groups below their model's cacheable-prefix minimum are excluded."),
        confidence="documented",
        evidence=ev,
        detail={"groups": len(ev)},
        row_savings=pd.concat(savings_parts) if savings_parts else pd.Series(dtype=float),
    )


# ---------------------------------------------------------------------------
# 2. Model over-provisioning
# ---------------------------------------------------------------------------

def detect_right_sizing(df: pd.DataFrame, rc: RateCard, days: float) -> Optional[Finding]:
    th = rc.thresholds.get("right_sizing", {})
    max_out = int(th.get("max_output_tokens", 200))
    max_in = int(th.get("max_input_tokens", 8000))
    min_calls = int(th.get("min_calls", 25))

    cand = df[df.priced & df.tier.isin(["frontier", "mid"])]
    if cand.empty:
        return None

    rows: list[dict] = []
    savings_parts: list[pd.Series] = []
    for (agent, provider, model), g in cand.groupby(["agent", "provider", "model"]):
        target = rc.downgrade_target(str(provider), str(model))
        if not target:
            continue
        t_name, t_entry = target

        # "Easy" work: short answers over short context. Long context may
        # genuinely need the bigger model, so those rows are left alone.
        easy = g[(g.output_tokens.fillna(0) <= max_out)
                 & (g.total_prompt_tokens <= max_in)]
        if len(easy) < min_calls:
            continue
        share = len(easy) / len(g)

        scale_in = float(t_entry["input"]) / max(float(g["base_input_rate"].iloc[0]), 1e-9)
        scale_out = float(t_entry["output"]) / max(float(g["base_output_rate"].iloc[0]), 1e-9)

        per_row_new = (
            (easy["cost_input"] + easy["cost_cache_read"] + easy["cost_cache_write"]) * scale_in
            + easy["cost_output"] * scale_out
        )
        per_row_saving = (easy["cost_total"] - per_row_new).clip(lower=0)

        current = float(easy["cost_total"].sum())
        downgraded = float(per_row_new.sum())
        saving = current - downgraded
        if saving <= 0:
            continue

        rows.append({
            "agent": agent, "model": model, "suggested": t_name,
            "easy_calls": len(easy), "share_of_agent": round(share, 3),
            "median_output_tokens": int(easy["output_tokens"].median()),
            "paid": current, "if_downgraded": downgraded, "recoverable": saving,
        })
        savings_parts.append(
            pd.Series(per_row_saving.to_numpy(), index=easy["request_id"].to_numpy())
        )

    if not rows:
        return None

    ev = pd.DataFrame(rows).sort_values("recoverable", ascending=False)
    total = float(ev["recoverable"].sum())
    worst = ev.iloc[0]

    return Finding(
        key="right_sizing",
        title="Frontier models doing budget-model work",
        headline=(
            f"{worst['agent']} runs {int(worst['easy_calls']):,} calls on "
            f"{worst['model']} with a median output of "
            f"{int(worst['median_output_tokens'])} tokens."
        ),
        recoverable=total,
        monthly=to_monthly(total, days),
        calls_affected=int(ev["easy_calls"].sum()),
        agents=ev["agent"].drop_duplicates().tolist(),
        fix=("Route the short-output path to the smaller model and hold the "
             "frontier model for calls that actually need it. Validate with an "
             "eval set before switching — this is the one finding here that can "
             "cost you quality."),
        basis=("Calls with output ≤ {o} tokens and prompt ≤ {i} tokens re-priced "
               "at the next tier down in the rate card. Longer-context calls "
               "excluded.".format(o=max_out, i=max_in)),
        confidence="estimated",
        evidence=ev,
        row_savings=pd.concat(savings_parts) if savings_parts else pd.Series(dtype=float),
    )


# ---------------------------------------------------------------------------
# 3. Dead spend
# ---------------------------------------------------------------------------

def detect_dead_spend(df: pd.DataFrame, rc: RateCard, days: float) -> Optional[Finding]:
    errs = df[(df.status.astype("string").str.lower() == "error") & df.priced]
    trunc = df[(df.finish_reason.astype("string").str.lower() == "length") & df.priced]
    if errs.empty and trunc.empty:
        return None

    err_cost = float(errs["cost_total"].sum())
    trunc_cost = float(trunc["cost_total"].sum())
    total = err_cost + trunc_cost
    if total <= 0:
        return None

    parts = []
    if not errs.empty:
        parts.append(
            errs.groupby(errs["error_code"].fillna("(unknown)"))
            .agg(calls=("request_id", "size"), cost=("cost_total", "sum"))
            .reset_index().rename(columns={"error_code": "reason"})
        )
    if not trunc.empty:
        parts.append(pd.DataFrame([{
            "reason": "truncated (finish_reason=length)",
            "calls": len(trunc), "cost": trunc_cost,
        }]))
    ev = pd.concat(parts, ignore_index=True).sort_values("cost", ascending=False)

    err_rate = len(errs) / max(len(df), 1)
    return Finding(
        key="dead_spend",
        title="Spend on calls that produced nothing usable",
        headline=(
            f"{len(errs):,} failed calls ({err_rate:.1%} of traffic) and "
            f"{len(trunc):,} truncated generations were billed."
        ),
        recoverable=total,
        monthly=to_monthly(total, days),
        calls_affected=len(errs) + len(trunc),
        agents=pd.concat([errs["agent"], trunc["agent"]]).drop_duplicates().tolist(),
        fix=("Rate-limit errors: add backoff and a concurrency cap. Truncations: "
             "raise max_tokens or shorten the requested output — a cut-off answer "
             "is usually retried, so you pay twice for one result."),
        basis=("Full billed cost of rows with status=error, plus rows with "
               "finish_reason=length whose output was cut mid-generation."),
        confidence="documented",
        evidence=ev,
        detail={"error_cost": err_cost, "truncation_cost": trunc_cost},
        row_savings=pd.concat([
            pd.Series(d["cost_total"].to_numpy(), index=d["request_id"].to_numpy())
            for d in (errs, trunc) if not d.empty
        ]).groupby(level=0).max(),
    )


# ---------------------------------------------------------------------------
# 4. Runaway agent loops
# ---------------------------------------------------------------------------

def detect_runaway_loops(df: pd.DataFrame, rc: RateCard, days: float) -> Optional[Finding]:
    th = rc.thresholds.get("runaway_loop", {})
    min_repeats = int(th.get("min_repeats", 10))

    cand = df.dropna(subset=["session_id", "prompt_full_hash"])
    cand = cand[cand.priced]
    if cand.empty:
        return None

    grp = cand.groupby(["session_id", "prompt_full_hash"])
    sizes = grp.size()
    loops = sizes[sizes >= min_repeats]
    if loops.empty:
        return None

    rows: list[dict] = []
    savings_parts: list[pd.Series] = []
    for (sess, phash), n in loops.items():
        g = grp.get_group((sess, phash))
        # One retry is legitimate engineering. Everything past the second
        # identical call is burn.
        wasted = float(g["cost_total"].sum()) * (n - 2) / n
        span = (g["ts"].max() - g["ts"].min()).total_seconds() / 60.0
        rows.append({
            "session_id": str(sess)[:12], "agent": g["agent"].iloc[0],
            "model": g["model"].iloc[0], "identical_calls": int(n),
            "minutes": round(span, 1), "burned": wasted,
        })
        savings_parts.append(_spread(g, wasted))

    ev = pd.DataFrame(rows).sort_values("burned", ascending=False)
    total = float(ev["burned"].sum())
    worst = ev.iloc[0]

    return Finding(
        key="runaway_loop",
        title="Runaway agent loops",
        headline=(
            f"{len(ev)} session(s) repeated an identical prompt. Worst: "
            f"{worst['agent']} called the same prompt {int(worst['identical_calls'])}× "
            f"in {worst['minutes']:.0f} minutes."
        ),
        recoverable=total,
        monthly=to_monthly(total, days),
        calls_affected=int(ev["identical_calls"].sum()),
        agents=ev["agent"].drop_duplicates().tolist(),
        fix=("Add a max-steps ceiling and a no-progress detector: if the prompt "
             "hash repeats twice, break the loop and escalate rather than paying "
             "for the same call again."),
        basis=(f"Sessions where an identical prompt hash recurs ≥{min_repeats} times. "
               "The first two calls are treated as legitimate; the remainder is "
               "counted as burn."),
        confidence="documented",
        evidence=ev,
        severity="critical",
        row_savings=pd.concat(savings_parts) if savings_parts else pd.Series(dtype=float),
    )


# ---------------------------------------------------------------------------
# 5. Batchable workloads
# ---------------------------------------------------------------------------

def detect_batchable(df: pd.DataFrame, rc: RateCard, days: float) -> Optional[Finding]:
    th = rc.thresholds.get("batchable", {})
    start = int(th.get("off_hours_start", 22))
    end = int(th.get("off_hours_end", 6))
    min_calls = int(th.get("min_calls", 50))

    cand = df[df.priced & ~df.is_batch.fillna(False)]
    if cand.empty:
        return None

    hours = cand["ts"].dt.hour
    off = (hours >= start) | (hours <= end)
    cand = cand.assign(off_hours=off)

    rows: list[dict] = []
    savings_parts: list[pd.Series] = []
    for agent, g in cand.groupby("agent"):
        if len(g) < min_calls:
            continue
        share = float(g["off_hours"].mean())
        if share < 0.9:      # must be almost entirely off-hours to be safe
            continue
        cost = float(g["cost_total"].sum())
        mult = rc.multipliers(str(g["provider"].iloc[0]))["batch"]
        saving = cost * (1.0 - mult)
        if saving <= 0:
            continue
        rows.append({
            "agent": agent, "calls": len(g),
            "off_hours_share": round(share, 3),
            "paid": cost, "if_batched": cost * mult, "recoverable": saving,
        })
        savings_parts.append(pd.Series(
            (g["cost_total"] * (1.0 - mult)).to_numpy(), index=g["request_id"].to_numpy()))

    if not rows:
        return None

    ev = pd.DataFrame(rows).sort_values("recoverable", ascending=False)
    total = float(ev["recoverable"].sum())
    worst = ev.iloc[0]

    return Finding(
        key="batchable",
        title="Latency-tolerant work on the real-time endpoint",
        headline=(
            f"{worst['agent']} runs {int(worst['calls']):,} calls with "
            f"{worst['off_hours_share']:.0%} of traffic overnight, at full price."
        ),
        recoverable=total,
        monthly=to_monthly(total, days),
        calls_affected=int(ev["calls"].sum()),
        agents=ev["agent"].drop_duplicates().tolist(),
        fix=("Move these jobs to the Batch API. Nobody is waiting on a 1am "
             "backfill, and the discount is automatic — no prompt or model "
             "changes required."),
        basis=("Agents with ≥90% of calls between "
               f"{start}:00 and {end}:00 and no batch flag, re-priced at the "
               "documented batch multiplier."),
        confidence="documented",
        evidence=ev,
        row_savings=pd.concat(savings_parts) if savings_parts else pd.Series(dtype=float),
    )


# ---------------------------------------------------------------------------
# 6. Prompt bloat drift
# ---------------------------------------------------------------------------

def detect_prompt_bloat(df: pd.DataFrame, rc: RateCard, days: float) -> Optional[Finding]:
    th = rc.thresholds.get("prompt_bloat", {})
    min_growth = float(th.get("min_growth_pct", 25)) / 100.0
    min_weeks = int(th.get("min_weeks", 2))

    cand = df[df.priced].copy()
    if cand.empty or cand["ts"].isna().all():
        return None

    # Measured on TOTAL prompt tokens, not input_tokens. An agent that caches
    # its prefix shows bloat in cache_read, not input — reading input_tokens
    # alone silently misses drift in exactly the agents with the biggest
    # prefixes.
    cand["week"] = ((cand["ts"] - cand["ts"].min()).dt.days // 7).astype(int)

    rows: list[dict] = []
    savings_parts: list[pd.Series] = []
    for agent, g in cand.groupby("agent"):
        weekly = g.groupby("week")["total_prompt_tokens"].mean()
        if len(weekly) < min_weeks + 1:
            continue
        first, last = float(weekly.iloc[0]), float(weekly.iloc[-1])
        if first <= 0:
            continue
        growth = last / first - 1.0
        if growth < min_growth:
            continue

        # Recoverable = the excess above the starting size, priced at each
        # row's own blended input rate.
        excess = (g["total_prompt_tokens"] - first).clip(lower=0)
        input_cost = g["cost_input"] + g["cost_cache_read"] + g["cost_cache_write"]
        blended = input_cost / g["total_prompt_tokens"].replace(0, np.nan)
        per_row = (excess * blended).fillna(0)
        saving = float(per_row.sum())
        if saving <= 0:
            continue
        savings_parts.append(
            pd.Series(per_row.to_numpy(), index=g["request_id"].to_numpy()))

        rows.append({
            "agent": agent, "week_1_tokens": int(first), "latest_tokens": int(last),
            "growth": round(growth, 3), "calls": len(g), "recoverable": saving,
        })

    if not rows:
        return None

    ev = pd.DataFrame(rows).sort_values("recoverable", ascending=False)
    total = float(ev["recoverable"].sum())
    worst = ev.iloc[0]

    return Finding(
        key="prompt_bloat",
        title="System prompts growing week over week",
        headline=(
            f"{worst['agent']}'s prompt grew {worst['growth']:.0%} — from "
            f"{int(worst['week_1_tokens']):,} to {int(worst['latest_tokens']):,} tokens."
        ),
        recoverable=total,
        monthly=to_monthly(total, days),
        calls_affected=int(ev["calls"].sum()),
        agents=ev["agent"].drop_duplicates().tolist(),
        fix=("Audit what was appended. Prompt additions accumulate one sprint at "
             "a time and are almost never removed; most of the growth is rules "
             "that no longer apply."),
        basis=("Mean total prompt tokens per agent per week. Recoverable is the "
               "excess above week 1, priced at each row's blended input rate. "
               "Measured on total prompt tokens so cached prefixes are included."),
        confidence="estimated",
        evidence=ev,
        row_savings=pd.concat(savings_parts) if savings_parts else pd.Series(dtype=float),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

DETECTORS = (
    detect_runaway_loops,
    detect_cache_miss,
    detect_right_sizing,
    detect_batchable,
    detect_dead_spend,
    detect_prompt_bloat,
)


@dataclass
class Audit:
    findings: list[Finding]
    total_spend: float
    monthly_spend: float
    recoverable: float               # de-overlapped
    monthly_recoverable: float
    days: float
    priced_calls: int
    unpriced_calls: int
    naive_recoverable: float = 0.0   # what simply summing the findings claims
    row_combined: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def recoverable_share(self) -> float:
        return self.recoverable / self.total_spend if self.total_spend else 0.0

    @property
    def overlap_correction(self) -> float:
        return self.naive_recoverable - self.recoverable

    def by_agent(self, priced: pd.DataFrame) -> pd.DataFrame:
        """Cost Criminals: waste per agent, ranked — using de-overlapped
        savings, so the bars sum to the headline number."""
        if self.row_combined.empty:
            return pd.DataFrame(columns=["agent", "recoverable"])
        m = priced[["request_id", "agent"]].copy()
        m["recoverable"] = m["request_id"].map(self.row_combined).fillna(0.0)
        return (
            m.groupby("agent", as_index=False)["recoverable"].sum()
            .sort_values("recoverable", ascending=False)
        )

    def by_agent_finding(self) -> pd.DataFrame:
        """Standalone per-finding attribution, for the stacked breakdown."""
        rows: list[dict] = []
        for f in self.findings:
            if f.evidence.empty or "agent" not in f.evidence.columns:
                continue
            col = next((c for c in ("recoverable", "burned", "cost")
                        if c in f.evidence.columns), None)
            if col is None:
                continue
            for a, v in f.evidence.groupby("agent")[col].sum().items():
                rows.append({"agent": a, "finding": f.title, "recoverable": float(v)})
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["agent", "finding", "recoverable"])


def run_audit(priced: pd.DataFrame, rc: RateCard) -> Audit:
    days = period_days(priced)
    total_spend = float(priced["cost_total"].sum())

    findings: list[Finding] = []
    for fn in DETECTORS:
        try:
            f = fn(priced, rc, days)
        except Exception as exc:  # a broken detector must not kill the report
            print(f"[warn] {fn.__name__} failed: {exc}")
            continue
        if f is not None:
            findings.append(f)

    monthly_total = to_monthly(total_spend, days)
    for f in findings:
        if f.severity != "critical":
            f.severity = _severity(f.monthly, monthly_total)

    findings.sort(key=lambda f: f.recoverable, reverse=True)
    naive = float(sum(f.recoverable for f in findings))
    combined = _combine(findings, priced)
    recoverable = float(combined.sum())

    return Audit(
        findings=findings,
        total_spend=total_spend,
        monthly_spend=monthly_total,
        recoverable=recoverable,
        monthly_recoverable=to_monthly(recoverable, days),
        days=days,
        priced_calls=int(priced["priced"].sum()),
        unpriced_calls=int((~priced["priced"]).sum()),
        naive_recoverable=naive,
        row_combined=combined,
    )


def _combine(findings: list[Finding], priced: pd.DataFrame) -> pd.Series:
    """De-overlap savings across findings, per request.

    Findings are not disjoint: one nightly job can be flagged as both
    over-provisioned AND batchable. Summing the two claims more than that job
    ever cost. Levers compose multiplicatively on a given call — batching at
    0.5x on top of a model swap saves half of the already-reduced price, not
    half of the original — so the fractions are combined as

        combined = 1 - PROD(1 - fraction_i)

    which is bounded by the row's actual cost no matter how many findings name
    it.
    """
    cost = priced.set_index("request_id")["cost_total"].astype(float)
    cost = cost[cost > 0]
    if cost.empty:
        return pd.Series(dtype=float)

    remaining = pd.Series(1.0, index=cost.index)
    touched = pd.Series(False, index=cost.index)

    for f in findings:
        rs = f.row_savings
        if rs is None or len(rs) == 0:
            continue
        rs = rs.groupby(level=0).sum()
        rs = rs.reindex(cost.index).fillna(0.0)
        frac = (rs / cost).clip(lower=0.0, upper=1.0)
        remaining *= (1.0 - frac)
        touched |= frac > 0

    combined = cost * (1.0 - remaining)
    return combined[touched]
