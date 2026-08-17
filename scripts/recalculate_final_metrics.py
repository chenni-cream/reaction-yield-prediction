#!/usr/bin/env python3
"""Recalculate manuscript metrics from the committed final predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "model_training" / "results"

def calculate(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "rxntype", "true_yield", "pred_yield"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if predictions["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")
    values = predictions[["true_yield", "pred_yield"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain non-finite labels or estimates")
    rows = []
    for rxntype, group in predictions.groupby("rxntype", sort=True):
        y, p = group["true_yield"], group["pred_yield"]
        rows.append({"rxntype": str(int(rxntype)), "n_samples": len(group),
                     "r2": r2_score(y, p), "mae": mean_absolute_error(y, p),
                     "rmse": np.sqrt(mean_squared_error(y, p))})
    y, p = predictions["true_yield"], predictions["pred_yield"]
    rows.append({"rxntype": "pooled", "n_samples": len(predictions),
                 "r2": r2_score(y, p), "mae": mean_absolute_error(y, p),
                 "rmse": np.sqrt(mean_squared_error(y, p))})
    return pd.DataFrame(rows)

def main() -> None:
    predictions = pd.read_csv(RESULTS / "final_simple_ensemble_predictions.csv")
    expected = pd.read_csv(RESULTS / "final_simple_ensemble_metrics.csv", dtype={"rxntype": str})
    actual = calculate(predictions)
    merged = actual.merge(expected[["rxntype", "n_samples", "r2", "mae", "rmse"]],
                          on="rxntype", suffixes=("_actual", "_expected"), validate="one_to_one")
    for metric in ("r2", "mae", "rmse"):
        if not np.allclose(merged[f"{metric}_actual"], merged[f"{metric}_expected"], rtol=0, atol=1e-12):
            raise AssertionError(f"Recalculated {metric} does not match the committed metrics")
    class_rows = actual[actual["rxntype"] != "pooled"]
    pooled = actual[actual["rxntype"] == "pooled"].iloc[0]
    print(f"Samples: {len(predictions)}")
    print(f"Macro Test R²: {class_rows['r2'].mean():.6f}")
    print(f"Macro Test MAE: {class_rows['mae'].mean():.6f}")
    print(f"Pooled Test R²: {pooled['r2']:.6f}")
    print("Committed predictions and metrics agree.")

if __name__ == "__main__":
    main()
