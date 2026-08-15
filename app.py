"""
Token Burn Autopsy — Streamlit UI.

Layout follows the report, not the data: hero number, then the loop alarm if
there is one, then Cost Criminals, then findings with evidence. A user who
reads only the top of the page still leaves with the one number and the one
action.

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from engine import Audit, run_audit                      # noqa: E402
from normalize import ADAPTERS, normalize                # noqa: E402
from rates import DEFAULT_RATE_CARD, load_rate_card_from_yaml, price, unpriced_report  # noqa: E402
from schema import conform, describe, validate           # noqa: E402
from summary import action_list, audit_markdown, exec_summary, verdict   # noqa: E402

# --- palette (validated reference instance; light surface) -----------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"     # categorical slot 1 — recoverable
NEUTRAL = "#e1e0d9"      # irreducible spend, deliberately recessive
STATUS = {"critical": "#d03b3b", "serious": "#ec835a",
          "warning": "#fab219", "good": "#0ca30c"}

st.set_page_config(page_title="Token Burn Autopsy", page_icon="🔥", layout="wide")

st.markdown(f"""
<style>
  .hero {{ font-size: 3.4rem; font-weight: 700; line-height: 1.05;
           color: {INK}; letter-spacing: -0.02em; }}
  .hero-sub {{ font-size: 1.05rem; color: {INK_2}; margin-top: .15rem; }}
  .kicker {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .08em;
             color: {MUTED}; font-weight: 600; }}
  .pill {{ display:inline-block; padding: .12rem .55rem; border-radius: 999px;
           font-size: .72rem; font-weight: 600; border: 1px solid rgba(11,11,11,.10); }}
  .alarm {{ border-left: 4px solid {STATUS['critical']}; background: #fdf5f5;
            padding: .85rem 1.1rem; border-radius: 6px; margin: .5rem 0 1rem 0; }}
  .basis {{ font-size: .84rem; color: {INK_2}; border-left: 2px solid {GRID};
            padding-left: .7rem; }}
  .trust-note {{ border-left: 4px solid {STATUS['warning']}; background: #fff9ec;
                 padding: .75rem 1rem; border-radius: 6px; margin: .75rem 0; }}
  [data-testid="stMetricValue"] {{ font-size: clamp(1.65rem, 2.5vw, 2.35rem); white-space: nowrap; }}
  @media (max-width: 1100px) {{
    [data-testid="stMetricValue"] {{ font-size: 1.55rem; }}
  }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load(raw: bytes, name: str, adapter: str | None):
    df = pd.read_csv(pd.io.common.BytesIO(raw), low_memory=False)
    res = normalize(df, adapter_name=adapter)
    return res.df, res.adapter, res.notes, res.messages


@st.cache_data(show_spinner=False)
def _audit(df: pd.DataFrame, rc_yaml: str):
    rc = load_rate_card_from_yaml(rc_yaml)
    priced = price(conform(df), rc)
    return priced, run_audit(priced, rc), rc


def money(x: float) -> str:
    if abs(x) >= 100:
        return f"${x:,.0f}"
    if abs(x) >= 10:
        return f"${x:,.1f}"
    return f"${x:,.2f}"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_criminals(audit: Audit, priced: pd.DataFrame) -> go.Figure:
    """Waste per agent, ranked — with the rest of that agent's spend behind it.

    Two segments, so a legend is present. Recoverable takes the leading
    categorical hue; irreducible spend is deliberately recessive, because the
    story is the blue part.
    """
    waste = audit.by_agent(priced).set_index("agent")["recoverable"]
    spend = priced.groupby("agent")["cost_total"].sum()
    d = pd.DataFrame({"waste": waste, "spend": spend}).fillna(0.0)
    d["clean"] = (d["spend"] - d["waste"]).clip(lower=0)
    d = d.sort_values("waste")

    fig = go.Figure()
    fig.add_bar(
        y=d.index, x=d["waste"], orientation="h", name="Recoverable",
        marker=dict(color=SERIES_1, line=dict(color=SURFACE, width=2)),
        hovertemplate="<b>%{y}</b><br>Recoverable %{x:$,.2f}<extra></extra>",
    )
    fig.add_bar(
        y=d.index, x=d["clean"], orientation="h", name="Irreducible spend",
        marker=dict(color=NEUTRAL, line=dict(color=SURFACE, width=2)),
        hovertemplate="<b>%{y}</b><br>Irreducible %{x:$,.2f}<extra></extra>",
    )
    fig.update_layout(
        barmode="stack", height=60 + 46 * len(d),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=8, r=70, t=8, b=28),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    traceorder="normal", font=dict(color=INK_2, size=12)),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        xaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                   tickprefix="$", tickfont=dict(color=MUTED, size=11),
                   linecolor=BASELINE),
        yaxis=dict(showgrid=False, tickfont=dict(color=INK, size=13),
                   linecolor=BASELINE),
        bargap=0.35,
    )
    # Label past the end of the whole stack rather than at the end of the blue
    # segment — an in-stack label lands on the segment boundary and reads as if
    # it belongs to the grey.
    for agent, row in d.iterrows():
        share = row["waste"] / row["spend"] if row["spend"] else 0
        fig.add_annotation(
            x=row["spend"], y=agent, xanchor="left", yanchor="middle",
            xshift=8, showarrow=False,
            text=f"<b>{money(row['waste'])}</b>"
                 f"<span style='color:{MUTED}'> · {share:.0%}</span>",
            font=dict(size=12, color=INK),
        )
    return fig


def chart_bloat(priced: pd.DataFrame, agents: list[str]) -> go.Figure | None:
    """Mean prompt size per week for the agents flagged as bloating."""
    d = priced[priced.agent.isin(agents)].copy()
    if d.empty or d["ts"].isna().all():
        return None
    d["week"] = ((d["ts"] - d["ts"].min()).dt.days // 7).astype(int)
    piv = d.groupby(["week", "agent"])["total_prompt_tokens"].mean().unstack()

    fig = go.Figure()
    hues = [SERIES_1, "#eb6834", "#1baf7a", "#eda100"]
    for i, col in enumerate(piv.columns):
        fig.add_scatter(
            x=piv.index, y=piv[col], mode="lines+markers", name=str(col),
            line=dict(color=hues[i % len(hues)], width=2),
            marker=dict(size=8, line=dict(color=SURFACE, width=2)),
            hovertemplate="<b>%{fullData.name}</b><br>Week %{x}: "
                          "%{y:,.0f} tokens<extra></extra>",
        )
    fig.update_layout(
        height=300, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=8, r=8, t=8, b=28), hovermode="x unified",
        showlegend=len(piv.columns) > 1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=INK_2, size=12)),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        xaxis=dict(title=dict(text="Week", font=dict(color=MUTED, size=11)),
                   showgrid=False, tickfont=dict(color=MUTED, size=11),
                   linecolor=BASELINE, dtick=1),
        yaxis=dict(title=dict(text="Mean prompt tokens",
                              font=dict(color=MUTED, size=11)),
                   showgrid=True, gridcolor=GRID, zeroline=False,
                   tickfont=dict(color=MUTED, size=11), rangemode="tozero"),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Usage export")
SAMPLE_SOURCES = {
    "Sample data": Path("data/usage_canonical.csv"),
    "Sample data · Anthropic": Path("data/sample_anthropic.csv"),
    "Sample data · OpenAI": Path("data/sample_openai.csv"),
}
src = st.sidebar.radio(
    "Source", [*SAMPLE_SOURCES, "Upload my own"], label_visibility="collapsed")

adapter_names = ["auto-detect"] + [a.name for a in ADAPTERS]
chosen = st.sidebar.selectbox("Format", adapter_names, index=0)
adapter_arg = None if chosen == "auto-detect" else chosen

default_rc_text = DEFAULT_RATE_CARD.read_text()
rc_upload = st.sidebar.file_uploader(
    "Rate card (YAML)", type=["yaml", "yml"],
    help="Optional. Leave empty to price against the bundled rate_cards.yaml.")
st.sidebar.download_button(
    "Download rate_cards.yaml template", data=default_rc_text,
    file_name="rate_cards.yaml", mime="application/x-yaml")
rc_text = rc_upload.getvalue().decode("utf-8") if rc_upload is not None else default_rc_text

raw_bytes, source_name = None, ""
if src == "Upload my own":
    st.sidebar.info(
        "**Privacy note**\n\n"
        "This app has no database or external API calls: it analyses the CSV in "
        "the app session only. Still, this is a third-party hosted app—upload "
        "only data permitted by your organisation's policy."
    )
    up = st.sidebar.file_uploader("CSV export", type=["csv"])
    if up is not None:
        raw_bytes, source_name = up.getvalue(), up.name
    st.sidebar.caption(
        "Prompt text is never read — only token counts and hashes. "
        "Supported: OpenAI, Anthropic, OpenRouter, Langfuse, or canonical.")
else:
    sample = SAMPLE_SOURCES[src]
    if src == "Sample data" and not sample.exists():
        sample = Path("data/sample_usage.csv")
    if sample.exists():
        raw_bytes, source_name = sample.read_bytes(), sample.name
    else:
        st.sidebar.error(f"No sample found at {sample}. Run: python src/generate.py")
    if src != "Sample data":
        st.sidebar.caption(
            "A raw export in that provider's own column shape — Format below "
            "shows auto-detect recognising it, not the app's canonical schema.")

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.markdown('<div class="kicker">Token Burn Autopsy</div>', unsafe_allow_html=True)

if raw_bytes is None:
    st.title("Upload an LLM usage export")
    st.markdown(
        "This is not a spend dashboard. It returns **one number** — how much of "
        "your bill is recoverable — itemised into findings priced against "
        "published provider rate cards.\n\n"
        "Pick **Sample data** in the sidebar to see it run.")
    st.stop()

try:
    df, adapter_used, adapter_notes, msgs = _load(raw_bytes, source_name, adapter_arg)
except Exception as exc:
    st.error(f"Could not read that file.\n\n{exc}")
    st.stop()

try:
    priced, audit, rc = _audit(df, rc_text)
except Exception as exc:
    st.error(
        f"Could not use that rate card.\n\n{exc}\n\n"
        "Remove the upload in the sidebar to fall back to the bundled "
        "rate_cards.yaml, or download the template above and fix the file.")
    st.stop()
st.sidebar.caption(f"Adapter: **{adapter_used}** · {len(df):,} rows")
st.sidebar.caption(f"Active rate card dated {rc.verified_on}")
sources = rc.pricing_sources()
if sources:
    st.sidebar.caption("Provider pricing references")
    for provider, url in sources.items():
        st.sidebar.markdown(f"- [{provider.title()} docs]({url})")

head, qualifier = verdict(audit)

# --- hero ------------------------------------------------------------------
left, right = st.columns([3, 2])
with left:
    st.markdown(
        f'<div class="hero">{money(audit.monthly_recoverable)}<span '
        f'style="font-size:1.6rem;color:{INK_2};font-weight:500;"> / month '
        f'recoverable</span></div>'
        f'<div class="hero-sub">{head} — {audit.recoverable_share:.0%} of a '
        f'{money(audit.monthly_spend)}/month bill. {qualifier}</div>',
        unsafe_allow_html=True)
with right:
    a, b, c = st.columns(3)
    a.metric("Calls audited", f"{audit.priced_calls:,}")
    b.metric("Period", f"{audit.days:.0f} days")
    c.metric("Findings", len(audit.findings))

st.markdown("")

if any(f.confidence == "estimated" for f in audit.findings):
    st.markdown(
        f'<div class="trust-note"><b>Decision confidence.</b> '
        'Some opportunities are estimates, not guaranteed savings. Validate model or '
        'prompt changes against an eval set before production rollout.</div>',
        unsafe_allow_html=True)

# --- the alarm -------------------------------------------------------------
loops = next((f for f in audit.findings if f.key == "runaway_loop"), None)
if loops is not None and not loops.evidence.empty:
    w = loops.evidence.iloc[0]
    st.markdown(
        f'<div class="alarm"><b style="color:{STATUS["critical"]};">⚠ Runaway '
        f'loop detected</b> &nbsp;·&nbsp; <code>{w["agent"]}</code> called an '
        f'identical prompt <b>{int(w["identical_calls"])}×</b> in '
        f'{w["minutes"]:.0f} minutes — {money(w["burned"])} burned in one '
        f'session. {len(loops.evidence)} affected session(s) total.</div>',
        unsafe_allow_html=True)

tab_report, tab_findings, tab_quality, tab_schema = st.tabs(
    ["Report", "Findings & evidence", "Data quality", "Schema"])

# --- report ----------------------------------------------------------------
with tab_report:
    st.download_button(
        "Download audit as Markdown",
        data=audit_markdown(audit, rc.verified_on, sources),
        file_name="token-burn-audit.md",
        mime="text/markdown",
        help="A portable report with actions, methods, confidence, and pricing references.",
    )
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("#### Executive summary")
        st.markdown(exec_summary(audit))
    with c2:
        st.markdown("#### Cost criminals")
        st.caption("Recoverable waste per agent, against that agent's total spend.")
        st.plotly_chart(chart_criminals(audit, priced), width="stretch",
                        config={"displayModeBar": False})

    st.markdown("#### Do this first")
    acts = pd.DataFrame(action_list(audit))
    if not acts.empty:
        acts["monthly"] = acts["monthly"].map(money)
        acts["confidence"] = acts["confidence"].map(
            {"documented": "Provider-documented", "estimated": "Validate with eval"}
        )
        st.dataframe(
            acts.rename(columns={
                "rank": "#", "action": "Action", "finding": "Finding",
                "monthly": "Per month", "confidence": "Confidence",
                "agents": "Agents"}),
            hide_index=True, width="stretch")

# --- findings --------------------------------------------------------------
with tab_findings:
    if not audit.findings:
        st.success("No recoverable waste found by any of the six checks.")
    for f in audit.findings:
        colour = STATUS.get(
            {"critical": "critical", "high": "serious",
             "medium": "warning", "low": "good"}[f.severity], MUTED)
        with st.expander(
            f"{f.title}  —  {money(f.monthly)}/month  ·  {f.severity.upper()}",
            expanded=(f is audit.findings[0]),
        ):
            st.markdown(
                f'<span class="pill" style="color:{colour};">{f.severity}</span> '
                f'<span class="pill" style="color:{INK_2};">'
                f'{"validate with eval" if f.confidence == "estimated" else "provider-documented"}</span> '
                f'&nbsp; {f.calls_affected:,} calls affected',
                unsafe_allow_html=True)
            st.markdown(f"**{f.headline}**")
            st.markdown(f"**Fix.** {f.fix}")
            st.markdown(f'<div class="basis"><b>How this was calculated.</b> '
                        f'{f.basis}</div>', unsafe_allow_html=True)
            st.markdown("")
            if not f.evidence.empty:
                ev = f.evidence.copy()
                for col in ("paid", "if_cached", "if_downgraded", "if_batched",
                            "recoverable", "burned", "cost"):
                    if col in ev.columns:
                        ev[col] = ev[col].map(money)
                st.dataframe(ev, hide_index=True, width="stretch")
            if f.key == "prompt_bloat":
                fig = chart_bloat(priced, f.agents)
                if fig is not None:
                    st.plotly_chart(fig, width="stretch",
                                    config={"displayModeBar": False})

# --- data quality ----------------------------------------------------------
with tab_quality:
    st.markdown("#### How much of this file could be priced")
    q1, q2, q3 = st.columns(3)
    q1.metric("Priced calls", f"{audit.priced_calls:,}")
    q2.metric("Unpriced calls", f"{audit.unpriced_calls:,}")
    q3.metric("Naive vs de-overlapped",
              money(audit.recoverable),
              delta=f"-{money(audit.overlap_correction)} overlap",
              delta_color="off")

    st.caption(
        "Findings overlap: one call can be flagged by more than one detector. "
        "Summing them would claim more than the call ever cost, so savings are "
        "combined multiplicatively per request. The headline is the "
        "de-overlapped figure — the sum of the findings below it is not.")

    up = unpriced_report(priced)
    if not up.empty:
        st.markdown("**Models missing from the rate card** — excluded from all "
                    "totals rather than silently priced at zero.")
        st.dataframe(up, hide_index=True, width="stretch")

    warns = validate(conform(df))
    if warns:
        st.markdown("**Warnings**")
        for w in warns:
            st.markdown(f"- {w}")
    if msgs:
        st.markdown("**Adapter notes**")
        for m in msgs:
            st.markdown(f"- {m}")
    st.caption(f"Adapter used: {adapter_used}. {adapter_notes}")

# --- schema ----------------------------------------------------------------
with tab_schema:
    st.markdown("#### Canonical schema")
    st.caption(
        "Every provider export normalises onto this table. It stores "
        "observations only — there is no `is_wasteful` column, because a schema "
        "that hands the detector its answer makes the audit circular.")
    st.dataframe(describe(), hide_index=True, width="stretch")
