import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "model_training/results"


class FinalResultsTests(unittest.TestCase):
    def test_final_predictions(self):
        data = pd.read_csv(RESULTS / "final_simple_ensemble_predictions.csv")
        self.assertEqual(len(data), 10763)
        self.assertTrue(data["sample_id"].is_unique)
        self.assertEqual(sorted(data["rxntype"].unique().tolist()), list(range(1, 9)))
        self.assertTrue(np.isfinite(data[["true_yield", "pred_yield"]].to_numpy()).all())
        self.assertTrue(np.isclose(
            r2_score(data.true_yield, data.pred_yield),
            0.4390213422239878,
            atol=1e-12,
        ))

    def test_oof_analysis_schema(self):
        data = json.loads((RESULTS / "ensemble_oof_analysis.json").read_text())
        self.assertEqual(data["purpose"], "exploratory_analysis_only")
        self.assertEqual(data["final_manuscript_method"], "simple_average")
        self.assertEqual(sorted(map(int, data["reactions"])), list(range(1, 9)))


if __name__ == "__main__":
    unittest.main()
