"""Create Git-ignored private synthetic demo fixtures."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_growth.synthetic.generator import generate_dataset, write_dataset  # noqa: E402


if __name__ == "__main__":
    summary = write_dataset(generate_dataset(), ROOT / "data" / "generated")
    print(json.dumps(summary, indent=2))
