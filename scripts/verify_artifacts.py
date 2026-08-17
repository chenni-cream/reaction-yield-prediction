#!/usr/bin/env python3
"""Validate inference-only or complete pretrained artifact layouts."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REACTIONS = range(1, 9)
FOLDS = range(1, 6)
TRAIN_COUNTS = {1: 23538, 2: 14455, 3: 5423, 4: 3292, 5: 6919, 6: 10465, 7: 6820, 8: 25940}
MODEL_DIRECTORIES = (
    "Avalon_RDKit",
    "Layered_RDKit_QC_Solvent",
)

def require(path: Path, errors: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        errors.append(f"Missing or empty: {path.relative_to(ROOT)}")

def model_paths(avalon: Path, optuna: Path):
    for rxn in REACTIONS:
        for fold in FOLDS:
            yield avalon / f"rxn_{rxn}/lgbm_fold{fold}.txt"
            for name in MODEL_DIRECTORIES:
                yield optuna / name / f"rxn_{rxn}/lgbm_fold{fold}.txt"

def validate(mode: str = "full", load_models: bool = False) -> list[str]:
    if mode not in {"inference", "full"}:
        raise ValueError("mode must be 'inference' or 'full'")
    errors: list[str] = []
    avalon = ROOT / "model_training/ckpt-searchfp/AvalonFingerprint_lgbm"
    optuna = ROOT / "model_training/ckpt-optuna"
    models = list(model_paths(avalon, optuna))
    for path in models:
        require(path, errors)
    if mode == "full":
        for rxn in REACTIONS:
            require(optuna / f"Avalon_FP/rxn_{rxn}/oof_predictions.npy", errors)
            for name in MODEL_DIRECTORIES:
                directory = optuna / name / f"rxn_{rxn}"
                require(directory / "oof_predictions.npy", errors)
                require(directory / "feature_importance.csv", errors)
    if errors:
        return errors
    if mode == "full":
        for rxn in REACTIONS:
            for name in ("Avalon_FP",) + MODEL_DIRECTORIES:
                path = optuna / name / f"rxn_{rxn}/oof_predictions.npy"
                values = np.asarray(np.load(path)).reshape(-1)
                if len(values) != TRAIN_COUNTS[rxn]:
                    errors.append(f"Wrong OOF length: {path.relative_to(ROOT)} ({len(values)})")
                if not np.isfinite(values).all():
                    errors.append(f"Non-finite OOF values: {path.relative_to(ROOT)}")
            for name in MODEL_DIRECTORIES:
                path = optuna / name / f"rxn_{rxn}/feature_importance.csv"
                columns = set(pd.read_csv(path, nrows=1).columns)
                if not {"feature", "importance"}.issubset(columns):
                    errors.append(f"Invalid feature importance columns: {path.relative_to(ROOT)}")
    if load_models and not errors:
        import lightgbm as lgb
        for path in models:
            try:
                lgb.Booster(model_file=str(path))
            except Exception as exc:
                errors.append(f"Cannot load model {path.relative_to(ROOT)}: {exc}")
    return errors

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inference", "full"), default="full")
    parser.add_argument("--load-models", action="store_true", help="Load every required LightGBM model")
    args = parser.parse_args()
    errors = validate(args.mode, args.load_models)
    if errors:
        raise SystemExit("Artifact validation failed:\n  - " + "\n  - ".join(errors))
    print(f"All {args.mode} pretrained artifacts are complete and valid.")

if __name__ == "__main__":
    main()
