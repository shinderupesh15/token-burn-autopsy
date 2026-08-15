"""
Rate card loading and per-row pricing.

Everything downstream depends on this being right, so the module holds one
opinion firmly: a model that is not in the rate card is NOT priced at zero. It
is flagged and excluded from totals, and the exclusion is reported. A cost
audit that silently prices unknown rows at $0 understates the bill and tells
you the opposite of the truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

DEFAULT_RATE_CARD = Path(__file__).resolve().parents[1] / "rate_cards.yaml"
PER_MILLION = 1_000_000.0


@dataclass
class RateCard:
    raw: dict[str, Any]

    # -- lookups ------------------------------------------------------------
    @property
    def verified_on(self) -> str:
        return str(self.raw.get("meta", {}).get("verified_on", "unknown"))

    @property
    def currency(self) -> str:
        return str(self.raw.get("meta", {}).get("currency", "USD"))

    @property
    def thresholds(self) -> dict[str, Any]:
        return self.raw.get("thresholds", {})

    def multipliers(self, provider: str) -> dict[str, float]:
        blocks = self.raw.get("multipliers", {})
        block = blocks.get(str(provider).lower()) or blocks.get("default", {})
        return {
            "cache_read": float(block.get("cache_read", 1.0)),
            "cache_write_5m": float(block.get("cache_write_5m", 1.0)),
            "cache_write_1h": float(block.get("cache_write_1h", 1.0)),
            "batch": float(block.get("batch", 1.0)),
            "source": block.get("source", ""),
            "confidence": block.get("confidence", "assumed"),
        }

    def model(self, provider: str, model: str) -> Optional[dict[str, Any]]:
        models = self.raw.get("models", {})
        block = models.get(str(provider).lower(), {})
        entry = block.get(str(model))
        if entry:
            return entry
        # Fall back to a scan: OpenRouter rows may carry a provider we did not
        # anticipate, but the model name is still unambiguous.
        for prov_models in models.values():
            if str(model) in prov_models:
                return prov_models[str(model)]
        return None

    def known_models(self) -> set[str]:
        out: set[str] = set()
        for prov_models in self.raw.get("models", {}).values():
            out.update(prov_models.keys())
        return out

    def downgrade_target(self, provider: str, model: str) -> Optional[tuple[str, dict]]:
        entry = self.model(provider, model)
        if not entry:
            return None
        target = entry.get("downgrade_to")
        if not target:
            return None
        t_entry = self.model(provider, target)
        return (target, t_entry) if t_entry else None


def load_rate_card(path: str | Path = DEFAULT_RATE_CARD) -> RateCard:
    with open(path) as fh:
        return RateCard(yaml.safe_load(fh))


def load_rate_card_from_yaml(text: str) -> RateCard:
    """Parse a rate card from YAML text (e.g. an uploaded file's contents),
    rather than a filesystem path. A live deployment has no server filesystem
    a visitor can point a path at, so this is the loader the UI uses."""
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or "models" not in raw:
        raise ValueError("Not a valid rate card: missing top-level 'models' key.")
    return RateCard(raw)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def price(df: pd.DataFrame, rc: RateCard) -> pd.DataFrame:
    """Attach per-row cost columns.

    Adds: base_input_rate, base_output_rate, tier, batch_mult, cache_read_mult,
    cost_input, cost_output, cost_cache_read, cost_cache_write, cost_total,
    priced (bool).
    """
    out = df.copy()

    providers = out["provider"].astype("string").str.lower().fillna("other")
    models = out["model"].astype("string").fillna("")

    base_in, base_out, tiers, priced = [], [], [], []
    m_read, m_w5, m_w1, m_batch = [], [], [], []

    # Cache the per-(provider, model) lookup; logs repeat the same few pairs
    # tens of thousands of times.
    memo: dict[tuple[str, str], tuple] = {}
    for prov, mdl in zip(providers, models):
        key = (prov, mdl)
        if key not in memo:
            entry = rc.model(prov, mdl)
            mult = rc.multipliers(prov)
            memo[key] = (
                float(entry["input"]) if entry else 0.0,
                float(entry["output"]) if entry else 0.0,
                str(entry.get("tier", "unknown")) if entry else "unknown",
                entry is not None,
                mult["cache_read"], mult["cache_write_5m"],
                mult["cache_write_1h"], mult["batch"],
            )
        bi, bo, ti, pr, cr, w5, w1, bt = memo[key]
        base_in.append(bi); base_out.append(bo); tiers.append(ti); priced.append(pr)
        m_read.append(cr); m_w5.append(w5); m_w1.append(w1); m_batch.append(bt)

    out["base_input_rate"] = base_in
    out["base_output_rate"] = base_out
    out["tier"] = tiers
    out["priced"] = priced
    out["cache_read_mult"] = m_read
    out["batch_mult"] = [b if flag else 1.0
                         for b, flag in zip(m_batch, out["is_batch"].fillna(False))]

    write_mult = [
        w1 if str(ttl) == "1h" else w5
        for ttl, w5, w1 in zip(out["cache_write_ttl"].fillna(""), m_w5, m_w1)
    ]
    out["cache_write_mult"] = write_mult

    def toks(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce").fillna(0) / PER_MILLION

    out["cost_input"] = toks("input_tokens") * out["base_input_rate"] * out["batch_mult"]
    out["cost_output"] = toks("output_tokens") * out["base_output_rate"] * out["batch_mult"]
    out["cost_cache_read"] = (
        toks("cache_read_tokens") * out["base_input_rate"]
        * out["cache_read_mult"] * out["batch_mult"]
    )
    out["cost_cache_write"] = (
        toks("cache_write_tokens") * out["base_input_rate"]
        * out["cache_write_mult"] * out["batch_mult"]
    )
    out["cost_total"] = (
        out["cost_input"] + out["cost_output"]
        + out["cost_cache_read"] + out["cost_cache_write"]
    )

    # Unpriced rows contribute nothing and are reported separately.
    out.loc[~out["priced"], ["cost_input", "cost_output",
                            "cost_cache_read", "cost_cache_write", "cost_total"]] = 0.0

    out["total_prompt_tokens"] = (
        pd.to_numeric(out["input_tokens"], errors="coerce").fillna(0)
        + pd.to_numeric(out["cache_read_tokens"], errors="coerce").fillna(0)
        + pd.to_numeric(out["cache_write_tokens"], errors="coerce").fillna(0)
    )
    return out


def period_days(df: pd.DataFrame) -> float:
    if len(df) < 2 or df["ts"].isna().all():
        return 1.0
    span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0
    return max(span, 1.0)


def to_monthly(amount: float, days: float) -> float:
    """Normalise a period figure to a 30-day month."""
    return amount * (30.0 / days) if days > 0 else amount


def unpriced_report(df: pd.DataFrame) -> pd.DataFrame:
    bad = df[~df["priced"]]
    if bad.empty:
        return pd.DataFrame(columns=["provider", "model", "calls", "tokens"])
    return (
        bad.groupby(["provider", "model"])
        .agg(calls=("request_id", "size"), tokens=("total_prompt_tokens", "sum"))
        .reset_index()
        .sort_values("calls", ascending=False)
    )
