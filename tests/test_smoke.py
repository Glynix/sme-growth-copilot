"""Safe contracts and the merchant screen load without runtime errors."""

from dataclasses import fields
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from sme_growth.activation.adapter import MockActivationAdapter
from sme_growth.contracts import (
    AggregateOutcome,
    AudienceSummary,
    MIN_ELIGIBLE_AUDIENCE,
    MIN_OUTCOME_CONTRIBUTORS,
    MIN_PEER_MERCHANTS,
)


def test_safe_contracts_import_and_exclude_private_fields() -> None:
    assert (MIN_PEER_MERCHANTS, MIN_ELIGIBLE_AUDIENCE, MIN_OUTCOME_CONTRIBUTORS) == (
        10,
        50,
        10,
    )
    public_fields = {field.name for shape in (AudienceSummary, AggregateOutcome) for field in fields(shape)}
    forbidden_fields = {
        "synthetic_customer_id",
        "customer_id",
        "customer_token",
        "membership",
        "member_ids",
        "email",
        "phone",
        "individual_score",
    }
    assert public_fields.isdisjoint(forbidden_fields)
    assert MockActivationAdapter.__module__ == "sme_growth.activation.adapter"


def test_merchant_streamlit_app_renders() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Visa Versa Cafe"
    assert [tab.label for tab in app.tabs] == [
        "01 Overview", "02 Diagnose", "03 Recommend", "04 Activate", "05 Measure",
    ]
    tabs_column = next(column for column in app.columns if len(column.tabs) == 5)
    copilot_column = next(column for column in app.columns if len(column.chat_input) == 1)
    assert tabs_column.proto.weight == 0.7
    assert copilot_column.proto.weight == 0.3
    overview_cards = [item.value for item in app.markdown if 'class="overview-card"' in item.value]
    assert len(overview_cards) == 3
    assert any("Yesterday" in card and "AED 2,844" in card for card in overview_cards)
    assert any("Last 7 days" in card and "AED 17,234" in card for card in overview_cards)
    assert any("September 2026" in card and "AED 71,720" in card for card in overview_cards)
    assert all("Similar cafes:" in card for card in overview_cards)
    assert any("Daily net sales · September 2026" in item.value for item in app.markdown)
    assert any("median daily net sales across 36 similar cafes" in item.value for item in app.caption)
    trend = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert [trace["name"] for trace in trend["data"]] == ["Your sales", "Similar cafes median"]
    assert len(trend["data"][0]["x"]) == len(trend["data"][1]["x"]) == 30
    assert any(
        'class="peer-trend-card"' in item.value
        and 'class="peer-trend-footnote">Median of 36 similar merchants' in item.value
        for item in app.markdown
    )
    next(button for button in app.button if button.label == "⤢").click().run(timeout=30)
    assert not app.exception
    assert next(column for column in app.columns if len(column.tabs) == 5).proto.weight == 0.6
    assert next(column for column in app.columns if len(column.chat_input) == 1).proto.weight == 0.4
    next(button for button in app.button if button.label == "⤡").click().run(timeout=30)
    assert next(column for column in app.columns if len(column.tabs) == 5).proto.weight == 0.7
    assert next(column for column in app.columns if len(column.chat_input) == 1).proto.weight == 0.3
    assert any(metric.label == "Eligible lapsed regulars" and metric.value == "126" for metric in app.metric)
    assert any(metric.label == "Eligible at-risk regulars" and metric.value == "92" for metric in app.metric)
    assert [metric.value for metric in app.metric if metric.label == "Net Sales (AED)"] == ["711–1,497", "464–994"]
    assert sum("<div class=\"estimated\">Estimated scenario</div>" in item.value for item in app.markdown) == 2
    assert sum(item.label == "How the estimate was calculated" for item in app.expander) == 2
    assert any("Past spend is context, not predicted campaign sales." in item.value for item in app.markdown)
    assert any("spent about AED 5,976" in item.value and "per 30-day equivalent" in item.value for item in app.markdown)
    assert not any("Why am I seeing this?" in item.label for item in app.expander)
    assert app.selectbox[0].label == "Earlier four-week comparison"
    assert any(metric.label == "Net sales last week" for metric in app.metric)
    top_before = next(item.value for item in app.markdown if "Top opportunity:" in item.value)
    earlier_sales_before = [metric.value for metric in app.metric if metric.label == "Net sales"][1]
    app.selectbox[0].set_value(1).run(timeout=30)
    assert not app.exception
    assert next(item.value for item in app.markdown if "Top opportunity:" in item.value) == top_before
    assert [metric.value for metric in app.metric if metric.label == "Net sales"][1] != earlier_sales_before
