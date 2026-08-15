"""
Template-based executive summary.

Deliberately not an LLM call. Three reasons that survived the decision:

  1. A cost auditor that spends tokens to tell you about your token spend is a
     bad joke waiting to happen.
  2. Every sentence here is generated from a computed number. A model would
     paraphrase those numbers, and paraphrasing is exactly where a hallucinated
     figure enters a financial report.
  3. It runs offline, instantly, with no key — so the demo cannot fail live.

The writing quality comes from the templates being written once, carefully,
with the branching that a real analyst would apply.
"""

from __future__ import annotations

from engine import Audit


def _money(x: float) -> str:
    if x >= 100:
        return f"${x:,.0f}"
    if x >= 10:
        return f"${x:,.1f}"
    return f"${x:,.2f}"


def _pct(x: float) -> str:
    return f"{x:.0%}" if x >= 0.1 else f"{x:.1%}"


def verdict(audit: Audit) -> tuple[str, str]:
    """(headline verdict, one-line qualifier) based on how bad it is."""
    share = audit.recoverable_share
    if share >= 0.40:
        return ("Substantial waste", "This bill has never been optimised.")
    if share >= 0.20:
        return ("Meaningful waste", "A few structural fixes cover most of it.")
    if share >= 0.08:
        return ("Moderate waste", "The basics are in place; the edges are loose.")
    return ("Well optimised", "Little left on the table at current volume.")


def exec_summary(audit: Audit) -> str:
    """A short, plain-language brief. Markdown."""
    if not audit.findings:
        return (
            "**No recoverable waste found.**\n\n"
            f"Across {audit.priced_calls:,} priced calls over "
            f"{audit.days:.0f} days ({_money(audit.monthly_spend)}/month), none of "
            "the six checks found a lever worth pulling. Either this workload is "
            "already well tuned, or the export is missing the columns the "
            "detectors need — check the Data Quality tab before concluding the "
            "former."
        ).replace("$", r"\$")

    head, qualifier = verdict(audit)
    top = audit.findings[0]
    documented = [f for f in audit.findings if f.confidence == "documented"]
    doc_total = sum(f.recoverable for f in documented)

    lines: list[str] = []

    lines.append(
        f"**{head}: {_pct(audit.recoverable_share)} of this bill is recoverable.** "
        f"{qualifier}"
    )
    lines.append("")
    lines.append(
        f"Over {audit.days:.0f} days this workload cost {_money(audit.total_spend)} "
        f"({_money(audit.monthly_spend)}/month across {audit.priced_calls:,} priced "
        f"calls). {_money(audit.recoverable)} of that "
        f"({_money(audit.monthly_recoverable)}/month) is recoverable without "
        f"changing what any agent does."
    )

    # The overlap correction is worth stating out loud — it is the difference
    # between this report and one that just adds its findings together.
    if audit.overlap_correction > 0.005 * max(audit.total_spend, 1):
        lines.append("")
        lines.append(
            f"Note the findings below sum to {_money(audit.naive_recoverable)}, but "
            f"the recoverable total is {_money(audit.recoverable)}. The difference "
            f"({_money(audit.overlap_correction)}) is overlap: some calls are named "
            f"by more than one finding, and fixing them once does not pay twice. "
            f"The headline is the de-overlapped figure."
        )

    lines.append("")
    lines.append(f"**Biggest single lever — {top.title.lower()}.** {top.headline} "
                 f"Worth {_money(top.monthly)}/month. {top.fix}")

    if documented:
        lines.append("")
        lines.append(
            f"**{_money(doc_total)} of the total rests on published provider "
            f"pricing** — cache-read, batch and per-token rates you can verify "
            f"against the rate card. The remainder involves a judgement call and "
            f"is marked as such."
        )

    estimated = [f for f in audit.findings if f.confidence == "estimated"]
    if estimated:
        names = ", ".join(f.title.lower() for f in estimated)
        lines.append("")
        lines.append(
            f"**Validate before acting on: {names}.** These assume quality holds "
            f"after the change. Run an eval set before switching a model or "
            f"trimming a prompt in production."
        )

    critical = [f for f in audit.findings if f.severity == "critical"]
    loops = next((f for f in audit.findings if f.key == "runaway_loop"), None)
    if loops is not None:
        worst = loops.evidence.iloc[0]
        lines.append("")
        lines.append(
            f"**Fix first regardless of size: {int(worst['identical_calls'])} identical "
            f"calls from `{worst['agent']}` in {worst['minutes']:.0f} minutes.** "
            f"The cost is only {_money(loops.recoverable)}, but an agent that cannot "
            f"tell it is stuck is a reliability problem that happens to show up on "
            f"the invoice."
        )
    elif critical:
        lines.append("")
        lines.append(
            f"**Start with:** {', '.join(f.title.lower() for f in critical)} — "
            f"each accounts for over a fifth of monthly spend."
        )

    # Streamlit renders markdown with LaTeX enabled, so a pair of dollar signs
    # on one line is swallowed as math ("$845 ... $604" became an equation).
    # Escaping here keeps every figure literal.
    return "\n".join(lines).replace("$", r"\$")


def action_list(audit: Audit) -> list[dict]:
    """Ranked, one line each — what the reader does on Monday."""
    return [
        {
            "rank": i + 1,
            "action": f.fix.split(".")[0].strip() + ".",
            "finding": f.title,
            "monthly": f.monthly,
            "confidence": f.confidence,
            "agents": ", ".join(f.agents[:3]) + ("…" if len(f.agents) > 3 else ""),
        }
        for i, f in enumerate(audit.findings)
    ]


def audit_markdown(audit: Audit, verified_on: str, sources: dict[str, str]) -> str:
    """Create a portable, calculation-backed version of the current audit."""
    head, qualifier = verdict(audit)
    lines = [
        "# Token Burn Autopsy",
        "",
        f"## {_money(audit.monthly_recoverable)}/month recoverable",
        "",
        f"**{head}.** {audit.recoverable_share:.0%} of a "
        f"{_money(audit.monthly_spend)}/month bill. {qualifier}",
        "",
        "## Recommended actions",
        "",
        "| # | Action | Monthly opportunity | Confidence |",
        "|---:|---|---:|---|",
    ]
    for action in action_list(audit):
        confidence = ("Validate with an eval" if action["confidence"] == "estimated"
                      else "Provider-documented pricing")
        lines.append(
            f"| {action['rank']} | {action['action']} | {_money(action['monthly'])} | "
            f"{confidence} |"
        )

    lines.extend(["", "## Findings", ""])
    for finding in audit.findings:
        confidence = ("Estimate — validate with an eval before acting"
                      if finding.confidence == "estimated"
                      else "Provider-documented pricing")
        lines.extend([
            f"### {finding.title} — {_money(finding.monthly)}/month",
            "",
            f"- **Confidence:** {confidence}",
            f"- **Evidence:** {finding.headline}",
            f"- **Method:** {finding.basis}",
            f"- **Recommended fix:** {finding.fix}",
            "",
        ])

    lines.extend([
        "## Scope and provenance",
        "",
        f"- {audit.priced_calls:,} priced calls across {audit.days:.0f} day(s).",
        f"- The active rate card is dated **{verified_on}**.",
        "- Findings can overlap; the headline uses a per-request de-overlapped total.",
        "",
        "### Provider pricing references",
        "",
    ])
    lines.extend(f"- [{provider.title()}]({url})" for provider, url in sources.items())
    return "\n".join(lines)
