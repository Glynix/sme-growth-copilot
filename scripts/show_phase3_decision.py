"""Print safe model, audience, anomaly, and opportunity summaries only."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_growth.analytics import get_merchant_facts, get_peer_benchmark  # noqa: E402
from sme_growth.audiences.builder import PrivateAudienceStore, build_eligible_audience  # noqa: E402
from sme_growth.ml.anomaly import get_merchant_anomaly_signal  # noqa: E402
from sme_growth.ml.retention import get_retention_cohort_summary, train_retention_model  # noqa: E402
from sme_growth.recommendations.engine import generate_opportunities  # noqa: E402
from sme_growth.synthetic.generator import DEMO_MERCHANT_ID, GenerationConfig  # noqa: E402


if __name__ == "__main__":
    generated = ROOT / "data" / "generated"
    manifest = json.loads((generated / "run_summary.json").read_text(encoding="utf-8"))
    extension = manifest["format"]

    def load(name: str) -> pd.DataFrame:
        path = generated / f"{name}.{extension}"
        if extension == "parquet":
            return pd.read_parquet(path)
        if extension == "csv":
            return pd.read_csv(path, parse_dates={
                "merchants": ["opening_date"],
                "transactions": ["transaction_datetime"],
                "program_eligibility": ["valid_until", "revoked_at"],
            }[name])
        raise ValueError("Unsupported generated data format")

    transactions = load("transactions")
    merchants = load("merchants")
    program = load("program_eligibility")
    as_of = GenerationConfig().as_of_date
    facts = get_merchant_facts(transactions, merchants, program, DEMO_MERCHANT_ID, as_of)
    peers = get_peer_benchmark(transactions, merchants, DEMO_MERCHANT_ID, as_of)
    anomaly = get_merchant_anomaly_signal(transactions, DEMO_MERCHANT_ID, as_of, peers)
    model = train_retention_model(transactions, as_of)
    retention = get_retention_cohort_summary(model, transactions, DEMO_MERCHANT_ID, as_of)
    audience = build_eligible_audience(
        transactions, program, DEMO_MERCHANT_ID, as_of,
        PrivateAudienceStore(ROOT / "runtime" / "audiences.sqlite"),
    )
    opportunities = generate_opportunities(facts, audience, anomaly, retention)
    print(json.dumps({
        "model_evaluation": asdict(model.evaluation),
        "retention_cohort": asdict(retention),
        "anomaly": asdict(anomaly),
        "audience": asdict(audience),
        "opportunities": [asdict(item) for item in opportunities],
    }, indent=2, default=str))
