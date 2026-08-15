"""
Provider adapters -> canonical schema.

The subtle thing this module exists to handle is that providers disagree about
what "input tokens" means:

  Anthropic  input_tokens EXCLUDES cache_read_input_tokens (they are siblings)
  OpenAI     input_tokens INCLUDES cached tokens (cached_tokens is a SUBSET)

Get this wrong and you double-count cached input, which inflates every "you
could have saved $X" number in the report. Each adapter declares its convention
via `cache_is_subset` and the conversion happens in exactly one place.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from schema import conform, empty_frame  # noqa: F401


def _hash(text: object, length: int = 16) -> Optional[str]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text)
    if not s.strip():
        return None
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]


def _pick(df: pd.DataFrame, *candidates: str) -> Optional[pd.Series]:
    """First column present, matched case/separator-insensitively."""
    norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in norm:
            return df[norm[key]]
    return None


def _num(s: Optional[pd.Series], default: int = 0) -> pd.Series:
    if s is None:
        return pd.Series(default, dtype="Int64")
    return pd.to_numeric(s, errors="coerce").fillna(default).astype("Int64")


@dataclass
class Adapter:
    name: str
    provider: str
    cache_is_subset: bool
    detect: Callable[[pd.DataFrame], bool]
    build: Callable[[pd.DataFrame], pd.DataFrame]
    notes: str = ""


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def _canonical_detect(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return {"request_id", "input_tokens", "output_tokens", "model"} <= cols and "ts" in cols


def _canonical_build(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "provider" not in out.columns:
        out["provider"] = "other"
    return out


def _openai_detect(df: pd.DataFrame) -> bool:
    cols = {re.sub(r"[^a-z0-9]", "", c.lower()) for c in df.columns}
    has_tokens = {"inputtokens", "outputtokens"} <= cols or {"ninputtokens", "noutputtokens"} <= cols
    return has_tokens and bool({"cachedtokens", "inputcachedtokens", "ncachedtokens"} & cols)


def _openai_build(df: pd.DataFrame) -> pd.DataFrame:
    ts = _pick(df, "start_time", "timestamp", "ts", "date", "bucket_start")
    inp = _num(_pick(df, "input_tokens", "n_input_tokens", "prompt_tokens"))
    cached = _num(_pick(df, "cached_tokens", "input_cached_tokens", "n_cached_tokens"))
    out_tok = _num(_pick(df, "output_tokens", "n_output_tokens", "completion_tokens"))

    # OpenAI: cached is a SUBSET of input. Split it out so the canonical
    # `input_tokens` means "uncached input" everywhere downstream.
    uncached = (inp - cached).clip(lower=0)

    built = pd.DataFrame({
        "ts": ts,
        "request_id": _pick(df, "request_id", "id"),
        "session_id": _pick(df, "session_id", "conversation_id", "user_id"),
        "agent": _pick(df, "project_name", "project", "api_key_name", "endpoint", "operation"),
        "provider": "openai",
        "model": _pick(df, "model", "model_name"),
        "input_tokens": uncached,
        "output_tokens": out_tok,
        "cache_read_tokens": cached,
        "cache_write_tokens": 0,
        "cache_write_ttl": "",
        "status": _pick(df, "status") if _pick(df, "status") is not None else "ok",
        "error_code": _pick(df, "error_code", "error"),
        "finish_reason": _pick(df, "finish_reason", "stop_reason"),
        "latency_ms": _num(_pick(df, "latency_ms", "duration_ms"), 0),
        "is_batch": _pick(df, "batch", "is_batch"),
    })
    if built["request_id"].isna().all():
        built["request_id"] = [f"oai-{i:07d}" for i in range(len(built))]
    return built


def _anthropic_detect(df: pd.DataFrame) -> bool:
    cols = {re.sub(r"[^a-z0-9]", "", c.lower()) for c in df.columns}
    return bool({"cachereadinputtokens", "cachecreationinputtokens",
                 "cachereadtokens", "cachecreationtokens"} & cols)


def _anthropic_build(df: pd.DataFrame) -> pd.DataFrame:
    ts = _pick(df, "start_time", "timestamp", "ts", "date")
    # Anthropic: input_tokens already EXCLUDES cache reads. No subtraction.
    built = pd.DataFrame({
        "ts": ts,
        "request_id": _pick(df, "request_id", "id"),
        "session_id": _pick(df, "session_id", "conversation_id"),
        "agent": _pick(df, "workspace", "workspace_name", "api_key_name", "agent", "endpoint"),
        "provider": "anthropic",
        "model": _pick(df, "model", "model_name"),
        "input_tokens": _num(_pick(df, "input_tokens", "uncached_input_tokens")),
        "output_tokens": _num(_pick(df, "output_tokens")),
        "cache_read_tokens": _num(_pick(df, "cache_read_input_tokens", "cache_read_tokens")),
        "cache_write_tokens": _num(_pick(df, "cache_creation_input_tokens", "cache_creation_tokens")),
        "cache_write_ttl": _pick(df, "cache_ttl", "cache_write_ttl"),
        "status": _pick(df, "status") if _pick(df, "status") is not None else "ok",
        "error_code": _pick(df, "error_code", "error_type"),
        "finish_reason": _pick(df, "stop_reason", "finish_reason"),
        "latency_ms": _num(_pick(df, "latency_ms", "duration_ms"), 0),
        "is_batch": _pick(df, "is_batch", "batch"),
    })
    if built["request_id"].isna().all():
        built["request_id"] = [f"ant-{i:07d}" for i in range(len(built))]
    return built


def _openrouter_detect(df: pd.DataFrame) -> bool:
    cols = {re.sub(r"[^a-z0-9]", "", c.lower()) for c in df.columns}
    return "generationid" in cols or ({"tokensprompt", "tokenscompletion"} <= cols)


def _openrouter_build(df: pd.DataFrame) -> pd.DataFrame:
    model = _pick(df, "model", "model_permaslug")
    # OpenRouter models are namespaced 'anthropic/claude-...'; the prefix is
    # the most reliable provider signal available in the export.
    provider = (
        model.astype("string").str.split("/").str[0].str.lower()
        if model is not None else pd.Series("openrouter", index=df.index)
    )
    built = pd.DataFrame({
        "ts": _pick(df, "created_at", "timestamp", "ts", "date"),
        "request_id": _pick(df, "generation_id", "id"),
        "session_id": _pick(df, "session_id"),
        "agent": _pick(df, "app_name", "app", "api_key_name"),
        "provider": provider,
        "model": model.astype("string").str.split("/").str[-1] if model is not None else pd.NA,
        "input_tokens": _num(_pick(df, "tokens_prompt", "prompt_tokens", "native_tokens_prompt")),
        "output_tokens": _num(_pick(df, "tokens_completion", "completion_tokens",
                                    "native_tokens_completion")),
        "cache_read_tokens": _num(_pick(df, "native_tokens_cached", "cached_tokens")),
        "cache_write_tokens": 0,
        "cache_write_ttl": "",
        "status": _pick(df, "status"),
        "error_code": _pick(df, "error"),
        "finish_reason": _pick(df, "finish_reason", "native_finish_reason"),
        "latency_ms": _num(_pick(df, "generation_time", "latency_ms"), 0),
        "is_batch": False,
    })
    if built["request_id"].isna().all():
        built["request_id"] = [f"orr-{i:07d}" for i in range(len(built))]
    return built


def _langfuse_detect(df: pd.DataFrame) -> bool:
    cols = {re.sub(r"[^a-z0-9]", "", c.lower()) for c in df.columns}
    return "traceid" in cols or {"observationid", "tracename"} & cols != set()


def _langfuse_build(df: pd.DataFrame) -> pd.DataFrame:
    built = pd.DataFrame({
        "ts": _pick(df, "start_time", "timestamp", "created_at", "ts"),
        "request_id": _pick(df, "observation_id", "id", "generation_id"),
        # A Langfuse trace IS the agent run — the natural session boundary.
        "session_id": _pick(df, "session_id", "trace_id"),
        "agent": _pick(df, "trace_name", "name", "generation_name"),
        "provider": _pick(df, "provider", "model_provider"),
        "model": _pick(df, "model", "model_name"),
        "input_tokens": _num(_pick(df, "input_tokens", "prompt_tokens", "usage_input")),
        "output_tokens": _num(_pick(df, "output_tokens", "completion_tokens", "usage_output")),
        "cache_read_tokens": _num(_pick(df, "cache_read_input_tokens", "cached_tokens")),
        "cache_write_tokens": _num(_pick(df, "cache_creation_input_tokens")),
        "cache_write_ttl": "",
        "status": _pick(df, "status", "level"),
        "error_code": _pick(df, "status_message", "error"),
        "finish_reason": _pick(df, "finish_reason", "stop_reason"),
        "latency_ms": _num(_pick(df, "latency_ms", "duration_ms"), 0),
        "is_batch": False,
    })
    if built["provider"].isna().all():
        built["provider"] = "other"
    if built["request_id"].isna().all():
        built["request_id"] = [f"lfs-{i:07d}" for i in range(len(built))]
    return built


ADAPTERS: list[Adapter] = [
    Adapter("canonical", "any", False, _canonical_detect, _canonical_build,
            "Already in Token Burn Autopsy format."),
    Adapter("anthropic", "anthropic", False, _anthropic_detect, _anthropic_build,
            "input_tokens excludes cache reads — used as-is."),
    Adapter("openai", "openai", True, _openai_detect, _openai_build,
            "cached_tokens is a subset of input_tokens — subtracted out."),
    Adapter("openrouter", "openrouter", False, _openrouter_detect, _openrouter_build,
            "Provider inferred from the 'vendor/model' namespace."),
    Adapter("langfuse", "other", False, _langfuse_detect, _langfuse_build,
            "trace_id used as session_id when session_id is absent."),
]


@dataclass
class NormalizeResult:
    df: pd.DataFrame
    adapter: str
    notes: str
    dropped_rows: int = 0
    messages: list[str] = field(default_factory=list)


def detect_adapter(df: pd.DataFrame) -> Adapter:
    for ad in ADAPTERS:
        try:
            if ad.detect(df):
                return ad
        except Exception:
            continue
    raise ValueError(
        "Could not identify this export. Expected an OpenAI, Anthropic, "
        "OpenRouter or Langfuse usage export, or a file already in canonical "
        "format. Columns seen: " + ", ".join(map(str, df.columns[:20]))
    )


def normalize(df: pd.DataFrame, adapter_name: Optional[str] = None) -> NormalizeResult:
    """Normalise any supported export onto the canonical schema."""
    if adapter_name:
        matches = [a for a in ADAPTERS if a.name == adapter_name]
        if not matches:
            raise ValueError(f"Unknown adapter '{adapter_name}'.")
        adapter = matches[0]
    else:
        adapter = detect_adapter(df)

    built = adapter.build(df)
    messages: list[str] = []

    # Fingerprints. Real exports rarely ship prompt text; when they do, hash it
    # here so nothing sensitive travels further into the pipeline.
    if "prompt_prefix_hash" not in built.columns:
        prefix_src = _pick(df, "system_prompt", "system", "prompt_prefix", "instructions")
        full_src = _pick(df, "prompt", "input", "messages", "prompt_text")
        if prefix_src is not None:
            built["prompt_prefix_hash"] = prefix_src.map(_hash)
            messages.append("Derived prompt_prefix_hash by hashing the system prompt column.")
        elif full_src is not None:
            # Leading ~600 chars stands in for the stable prefix.
            built["prompt_prefix_hash"] = full_src.astype("string").str[:600].map(_hash)
            messages.append(
                "No system-prompt column; approximated the cacheable prefix from the "
                "first 600 characters of each prompt."
            )
        if full_src is not None and "prompt_full_hash" not in built.columns:
            built["prompt_full_hash"] = full_src.map(_hash)
            built["prompt_chars"] = full_src.astype("string").str.len()

    for col in ("prompt_prefix_hash", "prompt_full_hash", "prompt_chars"):
        if col not in built.columns:
            passthrough = _pick(df, col)
            built[col] = passthrough if passthrough is not None else pd.NA

    before = len(built)
    built = built[built["ts"].notna() | built["ts"].isna()]  # keep all; ts checked in validate
    out = conform(built, source=adapter.name)

    # A row with no billable tokens is noise (heartbeats, cancelled requests).
    billable = (
        out["input_tokens"].fillna(0) + out["output_tokens"].fillna(0)
        + out["cache_read_tokens"].fillna(0) + out["cache_write_tokens"].fillna(0)
    )
    out = out[billable > 0].reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        messages.append(f"Dropped {dropped} row(s) with no billable tokens.")

    return NormalizeResult(
        df=out, adapter=adapter.name, notes=adapter.notes,
        dropped_rows=dropped, messages=messages,
    )
