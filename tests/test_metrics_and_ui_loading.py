import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hypertrade.artifacts import load_run_table
from hypertrade.ui.state import load_manifest, numeric_columns, safe_metric


class MetricsAndUiLoadingTests(unittest.TestCase):
    def test_numeric_columns_and_safe_metric(self) -> None:
        df = pd.DataFrame(
            [
                {"number": 1, "total_pnl": 12.5, "profit_factor": 1.2, "name": "a"},
                {"number": 2, "total_pnl": 15.0, "profit_factor": 1.3, "name": "b"},
            ]
        )
        cols = numeric_columns(df, excluded={"number"})
        self.assertEqual(set(cols), {"total_pnl", "profit_factor"})
        self.assertEqual(safe_metric(df, "total_pnl"), 12.5)

    def test_manifest_and_table_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "manifest.json").write_text(json.dumps({"benchmark_name": "demo", "run_id": "demo_1"}))
            pd.DataFrame([{"number": 1, "total_pnl": 1.0}]).to_parquet(run_dir / "candidate_shortlist.parquet", index=False)

            manifest = load_manifest(str(run_dir))
            shortlist = load_run_table(str(run_dir), "candidate_shortlist.parquet")

            self.assertEqual(manifest["benchmark_name"], "demo")
            self.assertEqual(manifest["run_id"], "demo_1")
            self.assertEqual(len(shortlist), 1)


if __name__ == "__main__":
    unittest.main()
