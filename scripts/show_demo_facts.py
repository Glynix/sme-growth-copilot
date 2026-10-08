"""Print only the allowlisted Visa Versa Cafe analytics response."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_growth.analytics import get_merchant_facts  # noqa: E402
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

    facts = get_merchant_facts(
        load("transactions"), load("merchants"), load("program_eligibility"),
        DEMO_MERCHANT_ID, GenerationConfig().as_of_date,
    )
    print(json.dumps(asdict(facts), indent=2))
