"""
Canonical usage-log schema for Token Burn Autopsy.

Every provider export (OpenAI, Anthropic, OpenRouter, Langfuse/LangSmith) is
normalised onto ONE table so the savings engine never has to know where a row
came from.

Design rule that matters: this schema stores only *observations*, never
conclusions. There is no `is_wasteful` column, no `should_cache` flag. Every
finding the engine reports has to be derived from raw fields, because a column
that hands the detector its answer makes the whole audit circular.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    required: bool
    description: str


COLUMNS: tuple[Column, ...] = (
    # --- identity -----------------------------------------------------------
    Column("ts", "datetime64[ns, UTC]", True,
           "Request start time, UTC. Drives drift + off-hours detection."),
    Column("request_id", "string", True,
           "Unique per API call."),
    Column("session_id", "string", False,
           "Groups calls in one agent run/conversation. Required for loop detection."),
    Column("agent", "string", False,
           "Logical caller: 'research_v2', 'summariser', '/api/chat'. The unit "
           "a Cost Criminal leaderboard ranks."),

    # --- what was called ----------------------------------------------------
    Column("provider", "string", True,
           "openai | anthropic | openrouter | google | other. Selects rate card."),
    Column("model", "string", True,
           "Rate-card key, e.g. 'claude-opus-5', 'gpt-5-mini'."),

    # --- token accounting ---------------------------------------------------
    # Convention: input_tokens counts UNCACHED input only. Cached input is in
    # cache_read_tokens. They do not overlap. Providers disagree on this, so
    # every adapter must state which convention it is converting FROM.
    Column("input_tokens", "Int64", True, "Uncached input tokens, billed at 1x."),
    Column("output_tokens", "Int64", True, "Generated tokens."),
    Column("cache_read_tokens", "Int64", False, "Cache hits, billed at 0.1x."),
    Column("cache_write_tokens", "Int64", False, "Cache writes, billed at 1.25x or 2x."),
    Column("cache_write_ttl", "string", False, "'5m' | '1h' | '' — sets the write multiplier."),

    # --- outcome ------------------------------------------------------------
    Column("status", "string", True, "ok | error"),
    Column("error_code", "string", False, "Provider error code when status='error'."),
    Column("finish_reason", "string", False,
           "stop | length | tool_use | content_filter | error. "
           "'length' means truncated — output was billed but may be unusable."),
    Column("latency_ms", "Int64", False, "Round-trip ms."),
    Column("is_batch", "boolean", False, "True if already submitted via a batch endpoint (0.5x)."),

    # --- prompt fingerprints ------------------------------------------------
    # Hashes, never prompt text. Lets the tool run on a real production log
    # without exfiltrating anything — which is the difference between a demo
    # and something a viewer would actually run on their own bill.
    Column("prompt_prefix_hash", "string", False,
           "Hash of the stable leading segment (system prompt + tools). "
           "Repeats here with no cache_read are the cache-miss signal."),
    Column("prompt_full_hash", "string", False,
           "Hash of the whole rendered prompt. Consecutive repeats inside one "
           "session are the runaway-loop signal."),
    Column("prompt_chars", "Int64", False, "Rendered prompt length; backs drift analysis."),
)

REQUIRED = tuple(c.name for c in COLUMNS if c.required)
OPTIONAL = tuple(c.name for c in COLUMNS if not c.required)
ALL_COLUMNS = tuple(c.name for c in COLUMNS)

DTYPES = {c.name: c.dtype for c in COLUMNS}

# Sensible fills so a sparse export does not crash the engine. A missing
# cache_read is 0 (you did not cache); a missing session_id is not invented.
DEFAULTS = {
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "cache_write_ttl": "",
    "error_code": "",
    "finish_reason": "stop",
    "is_batch": False,
    "session_id": pd.NA,
    "agent": "unattributed",
}


class SchemaError(ValueError):
    pass


def empty_frame() -> pd.DataFrame:
    """An empty, correctly-typed canonical frame."""
    return pd.DataFrame({c.name: pd.Series(dtype=c.dtype) for c in COLUMNS})


def conform(df: pd.DataFrame, *, source: str = "unknown") -> pd.DataFrame:
    """Coerce a partially-populated frame to the canonical schema.

    Raises SchemaError if a required column is missing, because guessing at
    token counts would silently corrupt every downstream dollar figure.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(
            f"[{source}] missing required column(s): {', '.join(missing)}. "
            f"Required: {', '.join(REQUIRED)}"
        )

    out = df.copy()

    for name, default in DEFAULTS.items():
        if name not in out.columns:
            out[name] = default
        else:
            out[name] = out[name].fillna(default)

    for name in ALL_COLUMNS:
        if name not in out.columns:
            out[name] = pd.NA

    out = out[list(ALL_COLUMNS)]

    for col in COLUMNS:
        try:
            if col.dtype.startswith("datetime"):
                s = pd.to_datetime(out[col.name], utc=True, errors="coerce")
                out[col.name] = s
            elif col.dtype == "Int64":
                out[col.name] = pd.to_numeric(out[col.name], errors="coerce").astype("Int64")
            elif col.dtype == "boolean":
                out[col.name] = out[col.name].astype("boolean")
            else:
                out[col.name] = out[col.name].astype("string")
        except Exception as exc:  # pragma: no cover
            raise SchemaError(f"[{source}] column '{col.name}' -> {col.dtype}: {exc}") from exc

    return out.sort_values("ts").reset_index(drop=True)


def validate(df: pd.DataFrame) -> list[str]:
    """Non-fatal data-quality warnings, surfaced in the UI.

    These exist because a real export is always messier than a generated one,
    and silently auditing a broken file is worse than saying so.
    """
    warnings: list[str] = []
    n = len(df)
    if n == 0:
        return ["File contains no rows."]

    if df["ts"].isna().any():
        warnings.append(f"{int(df['ts'].isna().sum())} row(s) have an unparseable timestamp.")

    neg = df[["input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"]].lt(0).any()
    for col, bad in neg.items():
        if bad:
            warnings.append(f"'{col}' contains negative values.")

    if df["session_id"].isna().mean() > 0.5:
        warnings.append(
            "Over half the rows have no session_id — runaway-loop detection will be "
            "limited to whatever sessions are present."
        )

    if df["prompt_prefix_hash"].isna().mean() > 0.5:
        warnings.append(
            "Most rows have no prompt_prefix_hash — cache-miss detection will fall back "
            "to (agent, model) grouping, which is coarser."
        )

    if (df["cache_read_tokens"].fillna(0) == 0).all():
        warnings.append(
            "No cache reads anywhere in this file. Either caching is off entirely "
            "(a finding in itself) or your export omits cache columns."
        )

    span_days = (df["ts"].max() - df["ts"].min()).days if n > 1 else 0
    if span_days < 7:
        warnings.append(
            f"File spans {span_days} day(s). Prompt-bloat drift needs ~2+ weeks to be meaningful."
        )

    return warnings


def describe() -> pd.DataFrame:
    """Schema as a table — rendered on the app's Schema tab."""
    return pd.DataFrame(
        [
            {"column": c.name, "type": c.dtype,
             "required": "yes" if c.required else "no", "meaning": c.description}
            for c in COLUMNS
        ]
    )
