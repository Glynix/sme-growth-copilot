"""Single-merchant Streamlit demo; only released aggregate objects enter this file."""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from html import escape
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sme_growth.activation.campaign import CampaignError, CampaignState  # noqa: E402
from sme_growth.analytics.contracts import OverviewPeriod  # noqa: E402
from sme_growth.ui.backend import (  # noqa: E402
    HISTORICAL_COMPARISON_ENDS,
    advance_campaign, approve_campaign, ask_copilot, default_campaign_definition,
    get_dashboard, get_historical_comparison, get_last_week_pulse,
    get_latest_campaign_state, review_campaign, submit_campaign,
)


st.set_page_config(page_title="SME Growth Copilot", page_icon="☕", layout="wide")
st.markdown("""
<style>
  .stApp { background: #f7f8f5; }
  .block-container { max-width: 1120px; padding-top: 2.5rem; padding-bottom: 4rem; }
  [data-testid="stHeader"] { background: rgba(247,248,245,.92); }
  h1, h2, h3 { color: #19313a; letter-spacing: -.035em; }
  h1 { font-size: clamp(2.3rem, 5vw, 3.3rem) !important; margin-bottom: .15rem !important; }
  h3 { margin-top: .35rem !important; padding-top: 0; font-size: 1.5rem !important; }
  [data-testid="stCaptionContainer"] { color: #63747a; }
  [data-testid="stTabs"] [role="tablist"] { gap: .25rem; border-bottom: 1px solid #dce7e2; }
  [data-testid="stTabs"] [role="tab"] { color: #587075; font-weight: 700;
    padding: .65rem .8rem !important; white-space: nowrap; }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #126b68; }
  .section-kicker { color: #126b68; font-size: .75rem; font-weight: 800; letter-spacing: .12em;
    text-transform: uppercase; margin-bottom: -.6rem; }
  .observed, .estimated { display: inline-block; border-radius: 999px; padding: .26rem .62rem;
    font-weight: 750; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; }
  .observed { background: #e7f3ef; color: #126b68; }
  .estimated { background: #fff0d9; color: #935b0a; }
  .insight { background: #193e42; color: #f8fbf9; border-radius: 16px;
    padding: 1.15rem 1.3rem; margin: .9rem 0 1.2rem; line-height: 1.55; }
  .insight strong { color: #bce5d6; }
  [data-testid="stMetric"], .peer-trend-card { background: #fff; border: 1px solid #e1e9e5; border-radius: 16px;
    padding: 1.1rem 1.15rem; min-height: 122px; box-shadow: 0 3px 14px rgba(27,55,53,.035); }
  [data-testid="stMetricLabel"] { overflow: visible; }
  [data-testid="stMetricLabel"] p, .peer-trend-label { color: #66797c; font-size: .86rem; font-weight: 650;
    white-space: normal !important; line-height: 1.3; }
  [data-testid="stMetricValue"], .peer-trend-value { color: #19313a; font-size: clamp(1.1rem, 1.5vw, 1.35rem) !important; }
  .peer-trend-value { font-weight: 600; margin-top: .35rem; }
  .peer-trend-footnote { color: #63747a; font-size: .7rem; line-height: 1.3; margin-top: .25rem; }
  .overview-card { background: #fff; border: 1px solid #e1e9e5; border-radius: 16px;
    padding: .9rem 1rem; min-height: 154px; box-shadow: 0 3px 14px rgba(27,55,53,.035); }
  .overview-label { color: #66797c; font-size: .8rem; font-weight: 700; }
  .overview-value { color: #19313a; font-size: clamp(1rem, 1.45vw, 1.3rem);
    font-weight: 700; margin: .28rem 0 .4rem; overflow-wrap: anywhere; }
  .overview-comparison { color: #19313a; font-size: .73rem; line-height: 1.45; }
  .overview-footnote { color: #718285; font-size: .68rem; line-height: 1.3; margin-top: .3rem; }
  .st-key-copilot_panel { background: #fff; border-radius: 18px; }
  .st-key-copilot_size { display: flex; justify-content: flex-end; }
  .st-key-copilot_size button { width: 1.9rem; min-width: 1.9rem; height: 1.9rem;
    min-height: 1.9rem; padding: 0 !important; font-size: 1rem; line-height: 1; }
  .copilot-heading { color: #19313a; font-size: 1.08rem; font-weight: 750; line-height: 1.3; }
  .copilot-intro { color: #66797c; font-size: .76rem; line-height: 1.4; margin: .3rem 0 .6rem; }
  .copilot-row { display: flex; width: 100%; margin: .55rem 0; }
  .copilot-user { justify-content: flex-end; }
  .copilot-assistant { justify-content: flex-start; }
  .copilot-message { max-width: 92%; min-width: 0; }
  .copilot-sender { color: #718285; font-size: .66rem; font-weight: 700; margin: 0 .35rem .2rem; }
  .copilot-user .copilot-sender { text-align: right; }
  .copilot-bubble { font-size: .81rem; line-height: 1.45; padding: .58rem .72rem;
    overflow-wrap: anywhere; }
  .copilot-user .copilot-bubble { background: #126b68; color: #fff;
    border-radius: 14px 14px 4px 14px; }
  .copilot-assistant .copilot-bubble { background: #edf3f0; color: #19313a;
    border: 1px solid #dce7e2; border-radius: 14px 14px 14px 4px; }
  .st-key-copilot_panel [data-testid="stChatInput"] textarea,
  .st-key-copilot_panel [data-testid="stChatInput"] textarea::placeholder {
    font-size: .78rem !important; }
  [data-testid="stForm"] { background: #fff; border: 1px solid #dce7e2; border-radius: 18px;
    padding: 1.2rem 1.3rem; box-shadow: 0 4px 18px rgba(27,55,53,.04); }
  [data-testid="stExpander"] { background: #fff; border: 1px solid #e1e9e5; border-radius: 12px; }
  .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: #126b68; border-color: #126b68; border-radius: 10px; font-weight: 700; }
  .stButton button, [data-testid="stFormSubmitButton"] button { border-radius: 10px; }
  hr { border-color: #dce6e2 !important; }
  @media (max-width: 640px) {
    .block-container { padding: 1.1rem .9rem 3rem; }
    [data-testid="stMetric"] { min-height: 0; }
  }
</style>
""", unsafe_allow_html=True)


def money(value: float | Decimal | None) -> str:
    return f"AED {value:,.0f}" if value is not None else "Unavailable (privacy or evidence threshold)"


def display_window(start: str, end_exclusive: str) -> str:
    return f"{start} to {date.fromisoformat(end_exclusive) - timedelta(days=1)}"


def copilot_bubble(message: str, sender: str) -> str:
    role = "copilot-user" if sender == "user" else "copilot-assistant"
    label = "You" if sender == "user" else "Copilot"
    content = escape(message).replace("\n", "<br>")
    return (
        f'<div class="copilot-row {role}"><div class="copilot-message">'
        f'<div class="copilot-sender">{label}</div>'
        f'<div class="copilot-bubble">{content}</div></div></div>'
    )


_PRIVATE_QUESTION_FRAGMENT = re.compile(r"(?i)\bsyn_c_[a-z0-9_-]+\b|[\w.+-]+@[\w.-]+|\b\d{9,}\b")


def question_for_history(question: str) -> str:
    return _PRIVATE_QUESTION_FRAGMENT.sub("[private detail]", question)


def overview_card(label: str, period: OverviewPeriod, comparison: str) -> str:
    value = money(period.net_sales_aed)
    cafe_change = f"{period.sales_change_pct:+.1f}%" if period.sales_change_pct is not None else "Unavailable"
    peer_change = (
        f"{period.peer_group_change_pct:+.1f}%" if period.peer_group_change_pct is not None else "Unavailable"
    )
    peer_note = f" · {period.peer_count} similar cafes" if period.peer_count is not None else ""
    return (
        f'<div class="overview-card"><div class="overview-label">{label}</div>'
        f'<div class="overview-value">{value}</div>'
        f'<div class="overview-comparison">You: {cafe_change}<br>Similar cafes: {peer_change}</div>'
        f'<div class="overview-footnote">{comparison}{peer_note}</div></div>'
    )


def _state_panel(state: CampaignState | None, definition) -> None:
    if state is None:
        st.caption("Review the exact campaign before any simulation can be approved.")
        return
    st.write(f"**Campaign version {state.campaign_version} · {state.status.replace('_', ' ').title()}**")
    st.caption("Simulation events: " + " → ".join(event.replace("_", " ") for event in state.events))
    if state.receipt is not None and state.receipt.status != "failed":
        st.success(state.receipt.message)
        st.caption("Submission is not acceptance, dispatch, delivery or a customer contact.")
    if state.status == "submission_unknown":
        st.warning("Simulator response was lost. Reconcile the saved request before continuing.")
        if st.button("Reconcile submission", key="reconcile"):
            try:
                state = submit_campaign(state.campaign_ref, state.campaign_version, definition)
                st.rerun()
            except CampaignError as error:
                st.error(str(error))
            except Exception:
                st.error("The local simulation could not be reconciled. No customer was contacted.")
    elif state.status in ("submitted", "accepted", "simulated_dispatched"):
        if st.button("Run simulated dispatch and results", type="primary", key="advance"):
            try:
                advance_campaign(state.campaign_ref, state.campaign_version, definition)
                st.rerun()
            except CampaignError as error:
                st.error(str(error))
            except Exception:
                st.error("The local simulation could not continue. No customer was contacted.")
    elif state.status == "blocked":
        st.warning("Eligibility or approval is no longer valid. Refresh and review a new audience version.")
    elif state.status == "failed":
        st.error("The simulator failed. No customer was contacted. Review a new campaign version to retry.")

def _outcome_panel(state: CampaignState | None) -> None:
    if state is None or state.status != "completed" or state.outcome is None:
        st.info("Simulated results will appear here after the approved campaign completes.")
        return
    outcome = state.outcome
    st.markdown('<div class="observed">Simulated outcome · descriptive</div>', unsafe_allow_html=True)
    if outcome.suppressed:
        st.info("Outcome unavailable: a result cell or its complement is below the 10-contributor privacy threshold. Unavailable does not mean zero.")
    else:
        a, b, c = st.columns(3)
        a.metric("Simulated purchasers", f"{outcome.purchasers:,}")
        b.metric("Campaign-associated sales", money(outcome.campaign_associated_sales_aed))
        c.metric("Simulated campaign cost", money(outcome.campaign_cost_aed))
    st.caption(outcome.attribution_note)
    st.caption("One whole-campaign aggregate is shown; no overlapping audience or date cuts are released.")


try:
    with st.spinner("Preparing synthetic merchant insights…"):
        view = get_dashboard()
except Exception:
    st.title("SME Growth Copilot")
    st.error("The synthetic demo could not load. Run the local data generator and refresh this page.")
    st.stop()

st.title(view.merchant_name)
st.caption(f"{view.merchant_size.title()} {view.merchant_category.lower()} · {view.merchant_location} · Data as of {view.facts.as_of_date}")
copilot_expanded = st.session_state.get("copilot_expanded", False)
tabs_column, copilot_column = st.columns([6, 4] if copilot_expanded else [7, 3], gap="large")
overview_tab, diagnose_tab, recommend_tab, activate_tab, measure_tab = tabs_column.tabs(
    ["01 Overview", "02 Diagnose", "03 Recommend", "04 Activate", "05 Measure"]
)

with overview_tab:
    st.markdown('<div class="section-kicker">01 / Sales at a glance</div>', unsafe_allow_html=True)
    st.subheader("Overview")
    st.caption("Observed synthetic net sales · completed periods only")
    day_column, week_column, month_column = st.columns(3)
    day_column.markdown(
        overview_card("Yesterday", view.overview.day, "vs same weekday last week"),
        unsafe_allow_html=True,
    )
    week_column.markdown(
        overview_card("Last 7 days", view.overview.week, "vs previous 7 days"),
        unsafe_allow_html=True,
    )
    month_start = date.fromisoformat(view.overview.month.current_window.start)
    month_column.markdown(
        overview_card(month_start.strftime("%B %Y"), view.overview.month, "vs previous calendar month"),
        unsafe_allow_html=True,
    )
    st.caption("Similar-cafe figures compare total sales for a matched aggregate peer group. No individual cafe is shown.")
    st.markdown(f"**Daily net sales · {month_start.strftime('%B %Y')}**")
    if view.overview.daily_trend is not None:
        chart = go.Figure(go.Scatter(
            x=[point.date for point in view.overview.daily_trend],
            y=[point.net_sales_aed for point in view.overview.daily_trend],
            name="Your sales",
            mode="lines+markers", line=dict(color="#126b68", width=2.5),
            marker=dict(color="#126b68", size=5), fill="tozeroy",
            fillcolor="rgba(18,107,104,.11)",
            hovertemplate="%{x|%d %b}: AED %{y:,.0f}<extra></extra>",
        ))
        if view.overview.peer_daily_trend is not None:
            chart.add_trace(go.Scatter(
                x=[point.date for point in view.overview.peer_daily_trend],
                y=[point.median_net_sales_aed for point in view.overview.peer_daily_trend],
                name="Similar cafes median", mode="lines",
                line=dict(color="#c88731", width=2.5, dash="dash"),
                hovertemplate="Peer median · %{x|%d %b}: AED %{y:,.0f}<extra></extra>",
            ))
        chart.update_layout(
            height=300, margin=dict(l=12, r=12, t=34, b=24),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=view.overview.peer_daily_trend is not None,
            legend=dict(orientation="h", y=1.13, x=0),
            font=dict(color="#19313a", size=11),
        )
        chart.update_xaxes(type="date", nticks=7, showgrid=False)
        chart.update_yaxes(tickprefix="AED ", tickformat=",.0f", rangemode="tozero", gridcolor="#e8eeea")
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
        if view.overview.peer_daily_trend is not None:
            st.caption(
                f"Dashed line: median daily net sales across {view.overview.peer_daily_trend_count} similar cafes. "
                "Peer percentages above use aggregate group totals."
            )
        else:
            st.caption("Peer median unavailable under the daily peer-group privacy threshold.")
        st.caption("Daily trends are shown only when every day and its complement meet the privacy threshold.")
    else:
        st.info("Daily trend unavailable because a day or its complement is below the privacy threshold.")

with diagnose_tab:
    st.markdown('<div class="section-kicker">02 / Understand the change</div>', unsafe_allow_html=True)
    st.subheader("Diagnose")
    st.markdown('<div class="observed">Observed · latest 28 complete days versus previous 28 days</div>', unsafe_allow_html=True)
    col_sales, col_lunch, col_peer = st.columns(3)
    with col_sales:
        st.metric(
            "Net sales", money(view.sales.current_sales_aed),
            f"{view.sales.sales_change_pct:+.1f}% vs prior period" if view.sales.sales_change_pct is not None else None,
        )
    with col_lunch:
        st.metric(
            "Weekday lunch transactions",
            f"{view.facts.weekday_lunch_change_pct:+.1f}%" if view.facts.weekday_lunch_change_pct is not None else "Unavailable",
        )
    with col_peer:
        if (view.peers.status == "available" and view.peers.peer_count is not None
                and view.peers.peer_count >= 10 and view.peers.median_sales_change_pct is not None):
            peer_label = "Peer cafe trend"
            peer_value = f"{view.peers.median_sales_change_pct:+.1f}%"
            peer_note = f"Median of {view.peers.peer_count} similar merchants"
        else:
            peer_label = "Comparable cafes"
            peer_value = "Unavailable"
            peer_note = "Peer group below the 10-merchant privacy or evidence threshold."
        st.markdown(
            f'<div class="peer-trend-card"><div class="peer-trend-label">{peer_label}</div>'
            f'<div class="peer-trend-value">{peer_value}</div>'
            f'<div class="peer-trend-footnote">{peer_note}</div></div>',
            unsafe_allow_html=True,
        )

    if view.facts.largest_driver:
        driver = view.facts.largest_driver.replace("_", " ")
        st.markdown(f'<div class="insight"><strong>What stands out</strong><br>{driver.capitalize()} weakened most. This comparison points to where to look; it does not prove the cause.</div>', unsafe_allow_html=True)
    else:
        st.info("Insufficient released evidence to identify a main contributor.")

    st.markdown("**How does your sales change compare?**")
    if (
        view.peers.status == "available"
        and view.peers.peer_count is not None
        and view.peers.peer_count >= 10
        and view.facts.sales_change_pct is not None
        and view.peers.median_sales_change_pct is not None
    ):
        cafe_change = view.facts.sales_change_pct
        peer_change = view.peers.median_sales_change_pct
        gap = cafe_change - peer_change
        if gap < 0:
            st.write(f"**Your cafe is {abs(gap):.1f} percentage points below the peer median.**")
        elif gap > 0:
            st.write(f"**Your cafe is {gap:.1f} percentage points above the peer median.**")
        else:
            st.write("**Your cafe matches the peer median.**")

        values = (cafe_change, peer_change)
        padding = max(2.0, (max(*values, 0) - min(*values, 0)) * .25)
        chart = go.Figure()
        chart.add_vline(x=0, line_color="#9AAEAB", line_width=1.5)
        for row, value, color, name in (
            (1, cafe_change, "#D96D57", "Visa Versa Cafe"),
            (0, peer_change, "#126B68", "Similar cafes · median"),
        ):
            chart.add_shape(
                type="line", x0=0, x1=value, y0=row, y1=row,
                line=dict(color=color, width=8),
            )
            chart.add_trace(go.Scatter(
                x=[value], y=[row], mode="markers+text", name=name,
                marker=dict(color=color, size=18),
                text=[f"{value:+.1f}%"],
                textposition="middle left" if value < 0 else "middle right",
                textfont=dict(size=15, color=color),
                hovertemplate=f"{name}: %{{x:+.1f}}%<extra></extra>",
            ))
        chart.update_layout(
            height=235, margin=dict(l=20, r=35, t=18, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, font=dict(color="#19313A"),
            xaxis=dict(
                title="Net sales change vs previous 28 days", ticksuffix="%",
                range=[min(*values, 0) - padding, max(*values, 0) + padding],
                showgrid=True, gridcolor="#E3EAE6", zeroline=False,
            ),
            yaxis=dict(
                tickvals=[1, 0], ticktext=["Visa Versa Cafe", "Similar cafes · median"],
                range=[-.6, 1.6], showgrid=False, zeroline=False, automargin=True,
            ),
        )
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
        st.caption(f"Observed changes across the same two 28-day periods. Peer figure is the median of {view.peers.peer_count} similar merchants; no individual competitor is shown.")
    else:
        st.info("Comparable cafe evidence is unavailable under the peer-group threshold.")

    with st.expander("View an earlier period"):
        st.caption("Choose from fixed, non-overlapping four-week comparison pairs. Historical results may differ from the latest trend.")
        period_index = st.selectbox(
            "Earlier four-week comparison",
            options=range(len(HISTORICAL_COMPARISON_ENDS)),
            format_func=lambda index: (
                f"{HISTORICAL_COMPARISON_ENDS[index] - timedelta(days=28)} to "
                f"{HISTORICAL_COMPARISON_ENDS[index] - timedelta(days=1)}"
            ),
        )
        try:
            earlier = get_historical_comparison(period_index)
            current = earlier.sales.meta.current_window
            baseline = earlier.sales.meta.baseline_window
            st.caption(
                f"Observed · {display_window(current.start, current.end_exclusive)} "
                f"versus {display_window(baseline.start, baseline.end_exclusive)}"
            )
            older_sales, older_lunch, older_peers = st.columns(3)
            older_sales.metric(
                "Net sales", money(earlier.sales.current_sales_aed),
                f"{earlier.sales.sales_change_pct:+.1f}% vs prior four weeks"
                if earlier.sales.sales_change_pct is not None else None,
            )
            older_lunch.metric(
                "Weekday lunch transactions",
                f"{earlier.weekday_lunch_change_pct:+.1f}%"
                if earlier.weekday_lunch_change_pct is not None else "Unavailable",
            )
            older_peers.metric(
                "Peer cafe trend",
                f"{earlier.peers.median_sales_change_pct:+.1f}%"
                if earlier.peers.median_sales_change_pct is not None else "Unavailable",
            )
            st.caption("Historical diagnosis only. Audience, recommendations and activation use the latest 28-day snapshot.")
        except Exception:
            st.info("This earlier comparison is unavailable.")

    with st.expander("Last complete week · sales pulse"):
        try:
            pulse = get_last_week_pulse()
            st.caption(
                f"Observed · {display_window(pulse.current_window.start, pulse.current_window.end_exclusive)}. "
                "Compared with this cafe's own sales only."
            )
            if pulse.status == "available":
                week_sales, week_change, four_week_change = st.columns(3)
                week_sales.metric("Net sales last week", money(pulse.current_sales_aed))
                week_change.metric("Vs previous week", f"{pulse.change_vs_previous_week_pct:+.1f}%")
                four_week_change.metric(
                    "Vs 4-week average",
                    f"{pulse.change_vs_previous_four_week_average_pct:+.1f}%",
                )
                st.caption(
                    f"Previous week: {display_window(pulse.previous_week_window.start, pulse.previous_week_window.end_exclusive)}. "
                    f"Four-week reference: {display_window(pulse.previous_four_weeks_window.start, pulse.previous_four_weeks_window.end_exclusive)} "
                    "(total divided by four). This pulse does not drive recommendations."
                )
            else:
                st.info("Weekly pulse unavailable because a source week does not meet the privacy or evidence threshold.")
        except Exception:
            st.info("Weekly pulse unavailable.")

with recommend_tab:
    st.markdown('<div class="section-kicker">03 / Choose a practical action</div>', unsafe_allow_html=True)
    st.subheader("Recommend")
    st.caption("Based on the latest 28 days versus the previous 28 days.")
    if view.opportunities:
        top = view.opportunities[0]
        impact = top.estimated_impact_range
        st.markdown(f"**Top opportunity:** {top.title}")
        st.markdown('<div class="estimated">Estimated scenario</div>', unsafe_allow_html=True)
        if top.eligible_count is not None:
            audience_box, a, b, c = st.columns(4)
            audience_box.metric(
                "Eligible lapsed regulars" if top.opportunity_id == "weekday_lunch_winback" else "Eligible people",
                f"{top.eligible_count:,}",
            )
        else:
            a, b, c = st.columns(3)
        a.metric("Additional sales (AED)", f"{money(impact.incremental_sales_low_aed).replace('AED ', '')}–{money(impact.incremental_sales_high_aed).replace('AED ', '')}")
        b.metric("Campaign cost (AED)", f"{money(impact.campaign_cost_low_aed).replace('AED ', '')}–{money(impact.campaign_cost_high_aed).replace('AED ', '')}")
        c.metric("Net Sales (AED)", f"{money(impact.net_additional_sales_low_aed).replace('AED ', '')}–{money(impact.net_additional_sales_high_aed).replace('AED ', '')}")
        with st.expander("How the estimate was calculated"):
            if top.historical_value_aed_per_month is not None:
                st.write(f"**Observed history:** This targetable group previously spent about {money(top.historical_value_aed_per_month)} per 30-day equivalent. That spend is context, not recoverable revenue.")
            st.write(
                f"**Scenario assumptions:** {impact.assumptions.response_rate_low:.0%}–{impact.assumptions.response_rate_high:.0%} incremental return, "
                f"{impact.assumptions.visits_per_response:g} visits per responder, "
                f"{money(impact.assumptions.spend_per_visit_aed)} per visit. "
                "Net sales here means additional sales minus estimated discounts on all qualifying redemptions. "
                "Other expenses are not modeled. "
                "The range is not a confidence interval."
            )
        model_offer = next((item for item in view.opportunities if item.opportunity_id == "model_retention_offer"), None)
        if model_offer is not None:
            model_impact = model_offer.estimated_impact_range
            st.markdown(f"**Second campaign option:** {model_offer.title}")
            st.caption(
                "Top 20% predicted non-return risk among active regulars. "
                "This audience is separate from the lapsed win-back group."
            )
            st.markdown('<div class="estimated">Estimated scenario</div>', unsafe_allow_html=True)
            model_audience_box, x, y, z = st.columns(4)
            model_audience_box.metric("Eligible at-risk regulars", f"{model_offer.eligible_count:,}")
            x.metric("Additional sales (AED)", f"{money(model_impact.incremental_sales_low_aed).replace('AED ', '')}–{money(model_impact.incremental_sales_high_aed).replace('AED ', '')}")
            y.metric("Campaign cost (AED)", f"{money(model_impact.campaign_cost_low_aed).replace('AED ', '')}–{money(model_impact.campaign_cost_high_aed).replace('AED ', '')}")
            z.metric("Net Sales (AED)", f"{money(model_impact.net_additional_sales_low_aed).replace('AED ', '')}–{money(model_impact.net_additional_sales_high_aed).replace('AED ', '')}")
            st.caption("Review the overall 10% off offer in Activate. Response and sales are assumptions; the retention model predicts non-return, not offer response. Review each campaign separately if you want both.")
            with st.expander("How the estimate was calculated"):
                if view.model_audience.historical_spend_aed_per_30d_equivalent is not None:
                    st.write(
                        f"**Observed history:** These {model_offer.eligible_count:,} eligible active regulars "
                        f"previously spent about {money(view.model_audience.historical_spend_aed_per_30d_equivalent)} "
                        "per 30-day equivalent, calculated from their prior 90 days of settled purchases. "
                        "Past spend is context, not predicted campaign sales."
                    )
                else:
                    st.write("**Observed history:** Historical spend is unavailable under the privacy or evidence threshold.")
                st.write(
                    f"**Scenario assumptions:** {model_impact.assumptions.response_rate_low:.0%}–"
                    f"{model_impact.assumptions.response_rate_high:.0%} incremental retention, "
                    f"{model_impact.assumptions.visits_per_response:g} visits per responder, "
                    f"{money(model_impact.assumptions.spend_per_visit_aed)} per visit, "
                    f"{model_impact.assumptions.incentive_rate:.0%} discount, and "
                    f"{model_impact.assumptions.all_qualifying_redemption_rate_low:.0%}–"
                    f"{model_impact.assumptions.all_qualifying_redemption_rate_high:.0%} qualifying redemptions. "
                    "Net sales here means additional sales minus estimated discounts on all qualifying redemptions. "
                    "Other expenses are not modeled. "
                    "The response rate is an assumption, not a model prediction or measured lift. "
                    "The range is not a confidence interval."
                )
    else:
        st.info("There is not enough safe, observed evidence to recommend a campaign.")

with activate_tab:
    st.markdown('<div class="section-kicker">04 / Approve the exact plan</div>', unsafe_allow_html=True)
    st.subheader("Activate")
    st.caption("Review the audience version, offer, dates, channel and budget. Approval applies only to that exact definition.")
    campaign_options = [item for item in view.opportunities if item.opportunity_id in ("weekday_lunch_winback", "model_retention_offer")]
    selected_id = st.selectbox(
        "Campaign to review",
        options=[item.opportunity_id for item in campaign_options],
        format_func=lambda value: next(item.title for item in campaign_options if item.opportunity_id == value),
        ) if campaign_options else None
    st.caption("Choose either campaign, or review and approve each separately to run both. Audiences are disjoint.")
    default = default_campaign_definition(view, selected_id) if selected_id is not None else None
    if default is None:
        st.info("No audience-backed campaign is ready for review.")
    else:
        latest = get_latest_campaign_state(default.opportunity_id)
        with st.form("campaign_review"):
            selected_opportunity = next(item for item in campaign_options if item.opportunity_id == selected_id)
            st.write(f"**Private audience:** Version {default.audience_version} · {selected_opportunity.eligible_count} eligible people")
            offer = st.selectbox(
                "Offer", (15, 10) if selected_id == "weekday_lunch_winback" else (10,),
                format_func=lambda value: (
                    f"{value}% off Mon–Thu lunch, 11:00–15:00"
                    if selected_id == "weekday_lunch_winback" else f"{value}% off all purchases, any day"
                ),
            )
            d1, d2 = st.columns(2)
            start = d1.date_input("Start date", value=default.start_date)
            end = d2.date_input("End date", value=default.end_date)
            st.text_input("Channel", value="issuer_app", disabled=True)
            budget = st.number_input("Maximum budget (AED)", min_value=0.0, value=float(default.budget_aed), step=50.0)
            confirmed = st.checkbox("I approve this exact offer, audience version, dates, channel and budget for simulation only.")
            review_clicked = st.form_submit_button("Review Campaign")
            approve_clicked = st.form_submit_button("Approve Campaign", type="primary", disabled=latest is None)
        current = replace(
            default, offer_percent=int(offer), start_date=start, end_date=end,
            budget_aed=Decimal(str(budget)),
        )
        if review_clicked:
            try:
                latest = review_campaign(current, selected_opportunity)
                st.rerun()
            except CampaignError as error:
                st.error(str(error))
            except Exception:
                st.error("The campaign could not be reviewed. No customer was contacted.")
        if approve_clicked and latest is not None:
            try:
                approved = approve_campaign(latest.campaign_ref, latest.campaign_version, current, explicit=confirmed)
                latest = submit_campaign(approved.campaign_ref, approved.campaign_version, current)
            except CampaignError as error:
                st.error(str(error))
            except Exception:
                st.error("The campaign could not be submitted. No customer was contacted.")
        if latest is not None:
            st.caption(
                f"Reviewed exact terms: {latest.definition.offer_terms}; "
                f"{latest.definition.start_date} to {latest.definition.end_date}; "
                f"{latest.definition.channel}; budget {money(latest.definition.budget_aed)}. "
                f"Estimated high cost {money(latest.estimated_cost_high_aed)}."
            )
        _state_panel(latest, current)

with measure_tab:
    st.markdown('<div class="section-kicker">05 / Review the result</div>', unsafe_allow_html=True)
    st.subheader("Measure")
    for campaign in campaign_options:
        st.markdown(f"**{campaign.title}**")
        _outcome_panel(get_latest_campaign_state(campaign.opportunity_id))

with copilot_column.container(border=True, key="copilot_panel"):
    heading_column, toggle_column = st.columns([4, 1], gap="small")
    heading_column.markdown('<div class="copilot-heading">Ask the Copilot</div>', unsafe_allow_html=True)
    if toggle_column.button(
        "⤡" if copilot_expanded else "⤢", key="copilot_size",
        help="Make Copilot smaller" if copilot_expanded else "Make Copilot larger",
    ):
        st.session_state["copilot_expanded"] = not copilot_expanded
        st.rerun()
    st.markdown('<div class="copilot-intro">Ask about sales or next steps. Aggregate answers only.</div>', unsafe_allow_html=True)
    history = st.session_state.get("copilot_history", [])
    with st.container(height=420 if copilot_expanded else 260, border=False, autoscroll=True):
        if not history:
            st.markdown(copilot_bubble("Hi! Ask me what changed or what to do next.", "assistant"), unsafe_allow_html=True)
        for turn in history:
            st.markdown(copilot_bubble(turn["topic"], "user"), unsafe_allow_html=True)
            st.markdown(copilot_bubble(turn["answer"].text, "assistant"), unsafe_allow_html=True)
    question = st.chat_input("Ask a question about your business")
    if question:
        try:
            answer = ask_copilot(question, view, previous_intent=st.session_state.get("copilot_last_intent"))
            if answer.intent is not None:
                st.session_state["copilot_last_intent"] = answer.intent
            topic = question_for_history(question)
            st.session_state["copilot_history"] = (history + [{"topic": topic, "answer": answer}])[-8:]
            st.rerun()
        except Exception:
            st.error("The Copilot could not answer safely. The campaign workflow is unaffected.")

st.divider()
st.caption("Synthetic data · Simulated activation · Public Visa/issuer patterns are assumptions to validate, not access or endorsement. No customers contacted.")
