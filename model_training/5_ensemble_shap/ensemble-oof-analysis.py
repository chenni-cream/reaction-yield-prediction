#!/usr/bin/env python3
"""Compare simple averaging with adaptive weights on training OOF predictions.

This exploratory analysis has no test-data loading path. Adaptive weights are
reported only as a comparison; the manuscript final model is the predefined
arithmetic mean evaluated by ``evaluate-final-simple-ensemble.py``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score


TARGET_COLUMN = "Yield"
MODEL_NAMES = ["Avalon", "Avalon+RDKit", "Layered+NonFP"]
SIMPLE_WEIGHTS = np.full(len(MODEL_NAMES), 1.0 / len(MODEL_NAMES))
RXN_NAMES = {
    1: "C–N", 2: "Suzuki", 3: "Heck", 4: "Diels-Alder",
    5: "SNAr", 6: "Sonogashira", 7: "Michael", 8: "Amidation",
}


def load_train_data(dataset_dir: Path) -> pd.DataFrame:
    df1 = pd.read_csv(dataset_dir / "round1_train_data.csv").copy()
    df2 = pd.read_csv(dataset_dir / "round2_train_data.csv").copy()
    if "rxntype" not in df1.columns:
        df1["rxntype"] = 1
    if "rxntype" not in df2.columns:
        df2["rxntype"] = 2
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    df["rxntype"] = pd.to_numeric(df["rxntype"], errors="raise").astype(int)
    return df


def optimize_weights(oof_preds: list[np.ndarray], y_true: np.ndarray) -> np.ndarray:
    """Find non-negative R²-optimal weights constrained to sum to one."""
    n_models = len(oof_preds)

    def objective(weights: np.ndarray) -> float:
        normalized = weights / (weights.sum() + 1e-12)
        combined = sum(normalized[i] * oof_preds[i] for i in range(n_models))
        return -float(r2_score(y_true, combined))

    result = minimize(
        objective,
        SIMPLE_WEIGHTS.copy(),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Weight optimization failed: {result.message}")
    return result.x / result.x.sum()


def load_oof_predictions(
    rxntype: int,
    oof_roots: list[Path],
    expected_length: int,
) -> list[np.ndarray]:
    predictions = []
    for model_name, root in zip(MODEL_NAMES, oof_roots):
        path = root / f"rxn_{rxntype}" / "oof_predictions.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing {model_name} OOF predictions: {path}")
        pred = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        if len(pred) != expected_length:
            raise ValueError(
                f"{model_name} rxn_{rxntype} OOF length {len(pred)} "
                f"does not match training labels {expected_length}"
            )
        if not np.isfinite(pred).all():
            raise ValueError(f"Non-finite values in {model_name} OOF predictions: {path}")
        predictions.append(pred)
    return predictions


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    model_training_dir = script_dir.parent
    dataset_dir = model_training_dir.parent / "data"
    results_dir = model_training_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    oof_roots = [
        model_training_dir / "ckpt-optuna" / "Avalon_FP",
        model_training_dir / "ckpt-optuna" / "Avalon_RDKit",
        model_training_dir / "ckpt-optuna" / "Layered_RDKit_QC_Solvent",
    ]

    train_df = load_train_data(dataset_dir)
    frozen = {
        "schema_version": 1,
        "purpose": "exploratory_analysis_only",
        "selection_data": "training_oof_only",
        "final_manuscript_method": "simple_average",
        "simple_average_weights": SIMPLE_WEIGHTS.tolist(),
        "model_order": MODEL_NAMES,
        "reactions": {},
    }
    all_true, all_simple, all_adaptive = [], [], []

    for rxntype, rxn_df in train_df.groupby("rxntype", sort=True):
        rxntype = int(rxntype)
        rxn_df = rxn_df.reset_index(drop=True)
        y_true = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float64)
        oof_preds = load_oof_predictions(rxntype, oof_roots, len(y_true))
        simple_pred = np.mean(np.vstack(oof_preds), axis=0)
        weights = optimize_weights(oof_preds, y_true)
        combined = sum(w * pred for w, pred in zip(weights, oof_preds))
        simple_r2 = float(r2_score(y_true, simple_pred))
        adaptive_r2 = float(r2_score(y_true, combined))

        all_true.append(y_true)
        all_simple.append(simple_pred)
        all_adaptive.append(combined)

        frozen["reactions"][str(rxntype)] = {
            "name": RXN_NAMES.get(rxntype, f"rxn_{rxntype}"),
            "n_samples": len(y_true),
            "simple_average_oof_r2": simple_r2,
            "adaptive_weights": weights.tolist(),
            "oof_r2_per_model": [float(r2_score(y_true, p)) for p in oof_preds],
            "adaptive_weighted_oof_r2": adaptive_r2,
            "adaptive_minus_simple_r2": adaptive_r2 - simple_r2,
        }
        print(
            f"rxn_{rxntype}: n={len(y_true)}, "
            f"simple R²={simple_r2:.6f}, adaptive R²={adaptive_r2:.6f}, "
            f"delta={adaptive_r2 - simple_r2:+.6f}, "
            f"weights={np.round(weights, 6).tolist()}"
        )

    reaction_results = list(frozen["reactions"].values())
    frozen["summary"] = {
        "n_samples": int(sum(item["n_samples"] for item in reaction_results)),
        "macro_simple_average_oof_r2": float(np.mean([
            item["simple_average_oof_r2"] for item in reaction_results
        ])),
        "macro_adaptive_weighted_oof_r2": float(np.mean([
            item["adaptive_weighted_oof_r2"] for item in reaction_results
        ])),
        "pooled_simple_average_oof_r2": float(r2_score(
            np.concatenate(all_true), np.concatenate(all_simple)
        )),
        "pooled_adaptive_weighted_oof_r2": float(r2_score(
            np.concatenate(all_true), np.concatenate(all_adaptive)
        )),
    }

    output_path = results_dir / "ensemble_oof_analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=2, ensure_ascii=False)
    print(f"\nExploratory OOF analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
