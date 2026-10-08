"""Clearly labelled deterministic question previews; no LLM or raw-data access."""

from __future__ import annotations

from sme_growth.ui.backend import DashboardView


QUESTIONS = (
    "Why are sales down?",
    "Is the market slowing too?",
    "Which customers are we losing?",
    "What should I do next?",
    "How can I reach them privately?",
)


def placeholder_answer(question: str, view: DashboardView) -> str:
    """Use only already released structured facts, never private rows or guesses."""

    facts = view.facts
    if question == QUESTIONS[0]:
        if facts.sales_change_pct is None or facts.weekday_lunch_change_pct is None:
            return "The released data is insufficient to explain the sales change."
        return (
            f"Observed net sales changed {facts.sales_change_pct:+.1f}% versus the prior equal period. "
            f"The largest observed contributor was {facts.largest_driver or 'unavailable'}. "
            f"Weekday lunch transactions changed {facts.weekday_lunch_change_pct:+.1f}%. "
            "These are contributors, not proven causes."
        )
    if question == QUESTIONS[1]:
        if view.peers.status != "available" or view.peers.median_sales_change_pct is None:
            return "The peer group is below the privacy or evidence threshold, so no benchmark is shown."
        return (
            f"The median change among {view.peers.peer_count} comparable merchants was "
            f"{view.peers.median_sales_change_pct:+.1f}% over the same period. "
            "That comparison does not prove why this cafe changed."
        )
    if question == QUESTIONS[2]:
        if facts.repeat_customer_change_pct is None:
            return "The repeat-customer trend is unavailable under the privacy rules."
        return (
            f"Observed repeat customers changed {facts.repeat_customer_change_pct:+.1f}%. "
            "Private behavior rules identify lapsed regulars, but no identities or member list are released."
        )
    if question == QUESTIONS[3]:
        if not view.opportunities:
            return "There is not enough released evidence for a recommendation yet."
        top = view.opportunities[0]
        impact = top.estimated_impact_range
        return (
            f"The top proposal is: {top.title}. Its assumed additional sales range is "
            f"AED {impact.incremental_sales_low_aed:,.0f}–{impact.incremental_sales_high_aed:,.0f}. "
            "Review the exact offer and budget before approving any simulation."
        )
    if question == QUESTIONS[4]:
        if not view.audience.activation_eligible:
            return "No safe, eligible audience is currently available for a simulated campaign."
        return (
            f"The private simulator found {view.audience.eligible_count} eligible people. "
            "It retains membership internally and returns only an opaque audience reference and aggregate count. "
            "No customer is contacted in this prototype."
        )
    return "This preview supports the listed questions only. Conversational AI is planned for a later phase."
