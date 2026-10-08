"""Exact approval, mock retry, private dispatch, and safe measurement tests."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from sme_growth.activation.adapter import ApprovedCampaign, LocalMockPartnerAdapter, RECEIPT_MESSAGE
from sme_growth.activation.campaign import (
    ALL_DAYS, ALL_PURCHASES_WINDOW, CampaignDefinition, CampaignError, CampaignService,
)
from sme_growth.analytics import get_merchant_facts, get_peer_benchmark
from sme_growth.audiences.builder import PrivateAudienceStore, build_eligible_audience, build_model_risk_audience
from sme_growth.measurement.simulator import simulate_aggregate_outcome
from sme_growth.ml.anomaly import get_merchant_anomaly_signal
from sme_growth.ml.retention import get_retention_cohort_summary, train_retention_model
from sme_growth.recommendations.engine import generate_opportunities
from sme_growth.synthetic.generator import DEMO_MERCHANT_ID, PrivateSyntheticDataset, generate_dataset
import sme_growth.ui.backend as ui_backend


AS_OF = date(2026, 10, 1)


@pytest.fixture(scope="module")
def dataset() -> PrivateSyntheticDataset:
    return generate_dataset()


def _setup(dataset: PrivateSyntheticDataset, path):
    audience = build_eligible_audience(
        dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF,
        PrivateAudienceStore(path),
    )
    facts = get_merchant_facts(
        dataset.transactions, dataset.merchants, dataset.program_eligibility,
        DEMO_MERCHANT_ID, AS_OF,
    )
    peers = get_peer_benchmark(dataset.transactions, dataset.merchants, DEMO_MERCHANT_ID, AS_OF)
    anomaly = get_merchant_anomaly_signal(dataset.transactions, DEMO_MERCHANT_ID, AS_OF, peers)
    opportunity = generate_opportunities(facts, audience, anomaly)[0]
    definition = CampaignDefinition(
        DEMO_MERCHANT_ID, opportunity.opportunity_id,
        audience.audience_id, audience.audience_version, 15,
        AS_OF + timedelta(days=1), AS_OF + timedelta(days=6),
        "issuer_app", Decimal("700"),
    )
    return audience, opportunity, definition


def _service(dataset: PrivateSyntheticDataset, path, *, program=None, adapter=None):
    return CampaignService(
        path, dataset.transactions,
        dataset.program_eligibility if program is None else program,
        DEMO_MERCHANT_ID, AS_OF, adapter=adapter,
    )


def test_exact_review_and_explicit_approval_are_version_bound(dataset, tmp_path) -> None:
    path = tmp_path / "runtime" / "demo.sqlite"
    _, opportunity, definition = _setup(dataset, path)
    service = _service(dataset, path)
    first = service.review(definition, opportunity)
    assert first.status == "reviewed" and first.campaign_version == 1
    assert service.review(definition, opportunity) == first
    with pytest.raises(CampaignError, match="Confirm"):
        service.approve(first.campaign_ref, 1, definition, explicit=False)
    changed = replace(definition, budget_aed=Decimal("750"))
    with pytest.raises(CampaignError, match="details changed"):
        service.approve(first.campaign_ref, 1, changed, explicit=True)
    approved = service.approve(first.campaign_ref, 1, definition, explicit=True)
    assert approved.status == "approved"
    assert approved.approval_ref is not None
    with pytest.raises(CampaignError, match="details changed"):
        service.submit(first.campaign_ref, 1, changed)
    revised = service.review(changed, opportunity)
    assert revised.campaign_version == 2 and revised.status == "reviewed"
    assert revised.definition_hash != first.definition_hash
    with pytest.raises(CampaignError, match="no longer current"):
        service.submit(first.campaign_ref, 1, definition)
    with pytest.raises(CampaignError, match="Exact campaign approval"):
        service.submit(revised.campaign_ref, 2, changed)
    assert service.approve(revised.campaign_ref, 2, changed, explicit=True).status == "approved"


def test_mock_adapter_rejects_direct_unapproved_or_mismatched_submission(dataset, tmp_path) -> None:
    path = tmp_path / "demo.sqlite"
    _, opportunity, definition = _setup(dataset, path)
    adapter = LocalMockPartnerAdapter(path)
    service = _service(dataset, path, adapter=adapter)
    reviewed = service.review(definition, opportunity)
    with sqlite3.connect(path) as connection:
        key = connection.execute(
            "SELECT idempotency_key FROM campaign_versions WHERE campaign_ref = ? AND campaign_version = 1",
            (reviewed.campaign_ref,),
        ).fetchone()[0]
    forged = ApprovedCampaign(
        reviewed.campaign_ref, 1, definition.audience_id, definition.audience_version,
        "approval_forged", reviewed.definition_hash, key,
    )
    with pytest.raises(ValueError, match="persisted approval"):
        adapter.submit(forged)
    approved = service.approve(reviewed.campaign_ref, 1, definition, explicit=True)
    with pytest.raises(ValueError, match="persisted approval"):
        adapter.submit(forged)
    assert approved.approval_ref != forged.approval_ref
    assert service.submit(reviewed.campaign_ref, 1, definition).status == "submitted"


def test_revocation_after_submission_blocks_dispatch_and_requires_new_version(dataset, tmp_path) -> None:
    path = tmp_path / "runtime" / "demo.sqlite"
    _, opportunity, definition = _setup(dataset, path)
    service = _service(dataset, path)
    reviewed = service.review(definition, opportunity)
    service.approve(reviewed.campaign_ref, 1, definition, explicit=True)
    submitted = service.submit(reviewed.campaign_ref, 1, definition)
    assert submitted.status == "submitted"
    revoked = dataset.program_eligibility.copy()
    revoked.loc[0, "revoked_at"] = pd.Timestamp(AS_OF)
    resumed = _service(dataset, path, program=revoked)
    blocked = resumed.advance(reviewed.campaign_ref, 1, definition)
    assert blocked.status == "blocked"
    assert "accepted" in blocked.events and "simulated_dispatched" not in blocked.events
    with sqlite3.connect(path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM simulated_purchase_events").fetchone()[0] if connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'simulated_purchase_events'"
        ).fetchone() else 0
    assert count == 0
    fresh = build_eligible_audience(dataset.transactions, revoked, DEMO_MERCHANT_ID, AS_OF, PrivateAudienceStore(path))
    assert fresh.audience_version == 2 and fresh.eligible_count == 125
    with pytest.raises(CampaignError, match="stale|audience changed|audience is no longer"):
        resumed.review(definition, opportunity)


def test_small_or_expiring_audience_cannot_be_reviewed(dataset, tmp_path) -> None:
    path = tmp_path / "demo.sqlite"
    _, opportunity, definition = _setup(dataset, path)
    program = dataset.program_eligibility.copy()
    program.loc[program.index[:77], "permission_state"] = "unknown"
    small = _service(dataset, path, program=program)
    with pytest.raises(CampaignError):
        small.review(definition, opportunity)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_versions").fetchone()[0] == 0

    expiring_path = tmp_path / "expiring.sqlite"
    _setup(dataset, expiring_path)
    expiring = dataset.program_eligibility.copy()
    expiring.loc[expiring.index[:126], "valid_until"] = pd.Timestamp(AS_OF)
    with pytest.raises(CampaignError, match="no longer eligible"):
        _service(dataset, expiring_path, program=expiring).review(definition, opportunity)


def test_unknown_submission_reconciles_and_retries_are_idempotent(dataset, tmp_path) -> None:
    path = tmp_path / "demo.sqlite"
    _, opportunity, definition = _setup(dataset, path)
    adapter = LocalMockPartnerAdapter(path, lose_next_response=True)
    service = _service(dataset, path, adapter=adapter)
    reviewed = service.review(definition, opportunity)
    service.approve(reviewed.campaign_ref, 1, definition, explicit=True)
    unknown = service.submit(reviewed.campaign_ref, 1, definition)
    assert unknown.status == "submission_unknown" and unknown.receipt is None
    restarted = _service(dataset, path)
    submitted = restarted.submit(reviewed.campaign_ref, 1, definition)
    assert submitted.status == "submitted" and submitted.receipt.message == RECEIPT_MESSAGE
    assert restarted.submit(reviewed.campaign_ref, 1, definition).receipt == submitted.receipt
    completed = restarted.advance(reviewed.campaign_ref, 1, definition)
    assert completed.status == "completed"
    assert restarted.advance(reviewed.campaign_ref, 1, definition) == completed
    assert completed.events == (
        "reviewed", "approved", "submission_unknown", "submitted",
        "accepted", "simulated_dispatched", "completed",
    )
    assert completed.outcome.purchasers == 18
    assert completed.outcome.campaign_associated_sales_aed == Decimal("2160")
    assert completed.outcome.campaign_cost_aed == Decimal("324")
    assert "not incremental lift" in completed.outcome.attribution_note
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM partner_submissions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM simulated_purchase_events").fetchone()[0] == 36
        assert connection.execute("SELECT COUNT(*) FROM campaign_events WHERE event_type = 'submitted'").fetchone()[0] == 1
    public = json.dumps(asdict(completed), default=str)
    for forbidden in ("syn_c_", "synthetic_customer_id", "membership", "individual_score", "email", "phone"):
        assert forbidden not in public


def test_simulated_partner_failure_has_no_dispatch_or_outcome(dataset, tmp_path) -> None:
    path = tmp_path / "demo.sqlite"
    _, opportunity, definition = _setup(dataset, path)
    service = _service(dataset, path, adapter=LocalMockPartnerAdapter(path, fail_next_submission=True))
    reviewed = service.review(definition, opportunity)
    service.approve(reviewed.campaign_ref, 1, definition, explicit=True)
    failed = service.submit(reviewed.campaign_ref, 1, definition)
    assert failed.status == "failed"
    assert failed.outcome is None
    assert "simulated_dispatched" not in failed.events


def test_small_outcome_and_related_totals_are_suppressed(dataset, tmp_path) -> None:
    path = tmp_path / "demo.sqlite"
    program = dataset.program_eligibility.copy()
    program.loc[program.index[:76], "revoked_at"] = pd.Timestamp(AS_OF)
    audience = build_eligible_audience(
        dataset.transactions, program, DEMO_MERCHANT_ID, AS_OF, PrivateAudienceStore(path),
    )
    assert audience.activation_eligible and audience.eligible_count == 50
    kwargs = dict(
        path=path, campaign_ref="camp_test", campaign_version=1,
        merchant_id=DEMO_MERCHANT_ID, audience_id=audience.audience_id,
        audience_version=audience.audience_version, eligible_count=50,
        reporting_start=datetime(2026, 10, 2), reporting_end=datetime(2026, 10, 8),
        offer_percent=15, spend_per_visit_aed=Decimal("60"),
    )
    outcome = simulate_aggregate_outcome(**kwargs)
    assert outcome.suppressed
    assert outcome.purchasers is None
    assert outcome.campaign_associated_sales_aed is None
    assert outcome.campaign_cost_aed is None
    assert simulate_aggregate_outcome(**kwargs) == outcome
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM simulated_purchase_events").fetchone()[0] == 14
        existing = {row[0] for row in connection.execute(
            "SELECT DISTINCT synthetic_customer_id FROM simulated_purchase_events"
        )}
        members = [row[0] for row in connection.execute(
            "SELECT synthetic_customer_id FROM audience_members WHERE audience_id = ? AND audience_version = ?",
            (audience.audience_id, audience.audience_version),
        )]
        for index, token in enumerate(token for token in members if token not in existing):
            if len(existing) >= 45:
                break
            connection.execute(
                "INSERT INTO simulated_purchase_events VALUES (?, ?, ?, ?, ?, ?)",
                (f"private_test_event_{index}", "camp_test", 1, token, 6000, 900),
            )
            existing.add(token)
        assert len(existing) == 45
    complement_suppressed = simulate_aggregate_outcome(**kwargs)
    assert complement_suppressed.suppressed
    assert complement_suppressed.purchasers is None
    assert complement_suppressed.campaign_associated_sales_aed is None
    assert complement_suppressed.campaign_cost_aed is None


def test_both_disjoint_campaigns_require_separate_approval_and_complete(dataset, tmp_path) -> None:
    path = tmp_path / "both.sqlite"
    lapsed, winback, winback_definition = _setup(dataset, path)
    model = train_retention_model(dataset.transactions, AS_OF)
    model_audience = build_model_risk_audience(
        model, dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF,
        PrivateAudienceStore(path),
    )
    facts = get_merchant_facts(
        dataset.transactions, dataset.merchants, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF,
    )
    peers = get_peer_benchmark(dataset.transactions, dataset.merchants, DEMO_MERCHANT_ID, AS_OF)
    anomaly = get_merchant_anomaly_signal(dataset.transactions, DEMO_MERCHANT_ID, AS_OF, peers)
    retention = get_retention_cohort_summary(model, dataset.transactions, DEMO_MERCHANT_ID, AS_OF)
    choices = generate_opportunities(facts, lapsed, anomaly, retention, model_audience)
    model_offer = next(item for item in choices if item.opportunity_id == "model_retention_offer")
    model_definition = replace(
        winback_definition, opportunity_id=model_offer.opportunity_id,
        audience_id=model_audience.audience_id, audience_version=model_audience.audience_version,
        offer_percent=10, offer_days=ALL_DAYS, offer_window=ALL_PURCHASES_WINDOW,
    )
    service = _service(dataset, path)
    assert model_definition.offer_terms == "10% off all purchases, any day"
    with pytest.raises(CampaignError, match="10% off all purchases"):
        service.review(replace(model_definition, offer_percent=15), model_offer)
    with pytest.raises(CampaignError, match="10% off all purchases"):
        service.review(replace(model_definition, offer_days=winback_definition.offer_days,
                               offer_window=winback_definition.offer_window), model_offer)
    reviewed_winback = service.review(winback_definition, winback)
    reviewed_model = service.review(model_definition, model_offer)
    assert reviewed_winback.campaign_ref != reviewed_model.campaign_ref
    service.approve(reviewed_winback.campaign_ref, 1, winback_definition, explicit=True)
    with pytest.raises(CampaignError, match="Exact campaign approval"):
        service.submit(reviewed_model.campaign_ref, 1, model_definition)
    service.approve(reviewed_model.campaign_ref, 1, model_definition, explicit=True)
    for reviewed, definition in ((reviewed_winback, winback_definition), (reviewed_model, model_definition)):
        service.submit(reviewed.campaign_ref, 1, definition)
        completed = service.advance(reviewed.campaign_ref, 1, definition)
        assert completed.status == "completed"
        assert not completed.outcome.suppressed
    assert service.latest_state("weekday_lunch_winback").status == "completed"
    assert service.latest_state("model_retention_offer").status == "completed"


def test_streamlit_click_path_reviews_approves_and_measures(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ui_backend, "STATE_PATH", tmp_path / "ui-demo.sqlite")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=60)
    assert not app.exception
    app.button[0].click().run(timeout=60)
    assert not app.exception
    reviewed = ui_backend.get_latest_campaign_state("weekday_lunch_winback")
    assert reviewed.status == "reviewed"
    assert not app.button[1].disabled
    app.checkbox[0].set_value(True)
    app.button[1].click().run(timeout=60)
    assert not app.exception
    submitted = ui_backend.get_latest_campaign_state("weekday_lunch_winback")
    assert submitted.status == "submitted"
    advance = next(button for button in app.button if button.label == "Run simulated dispatch and results")
    advance.click().run(timeout=60)
    assert not app.exception
    completed = ui_backend.get_latest_campaign_state("weekday_lunch_winback")
    assert completed.status == "completed"
    assert completed.outcome.purchasers == 18


def test_streamlit_model_offer_can_be_chosen_and_approved(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ui_backend, "STATE_PATH", tmp_path / "model-ui.sqlite")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=60)
    assert not app.exception
    campaign_choice = next(item for item in app.selectbox if item.label == "Campaign to review")
    campaign_choice.set_value("model_retention_offer").run(timeout=60)
    assert not app.exception
    assert next(item for item in app.selectbox if item.label == "Offer").value == 10
    next(button for button in app.button if button.label == "Review Campaign").click().run(timeout=60)
    assert not app.exception
    reviewed = ui_backend.get_latest_campaign_state("model_retention_offer")
    assert reviewed.status == "reviewed"
    assert reviewed.definition.offer_terms == "10% off all purchases, any day"
    app.checkbox[0].set_value(True)
    next(button for button in app.button if button.label == "Approve Campaign").click().run(timeout=60)
    assert not app.exception
    assert ui_backend.get_latest_campaign_state("model_retention_offer").status == "submitted"
    assert ui_backend.get_latest_campaign_state("weekday_lunch_winback") is None
