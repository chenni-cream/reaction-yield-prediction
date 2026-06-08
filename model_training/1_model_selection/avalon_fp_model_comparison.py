# Conduct baseline comparison using RF / XGB / CatBoost / LGBM models
# Train the round1+round2 data by classifying according to rxntype using AvalonFingerprint
# Perform five-fold cross-validation to obtain the evaluation results for each reaction type
import warnings
from pathlib import Path
from time import perf_counter

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, DataStructs, RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from rdkit.Avalon import pyAvalonTools

    HAS_AVALON = True
except ImportError:
    HAS_AVALON = False

RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048


# ──────────────────────────────────────
def bitvect_to_numpy(bitvect: DataStructs.ExplicitBitVect, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def avalon_fp(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    if not HAS_AVALON:
        raise ImportError("rdkit.Avalon 不可用，无法计算 AvalonFingerprint")
    fp = pyAvalonTools.GetAvalonFP(mol, n_bits)
    return bitvect_to_numpy(fp, n_bits)


def encode_smiles_column(smiles_series: pd.Series) -> np.ndarray:
    smiles = smiles_series.fillna("").astype(str).tolist()
    rows = []
    zero = np.zeros((FP_SIZE,), dtype=np.uint8)
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(zero)
        else:
            rows.append(avalon_fp(mol, FP_SIZE))
    return np.asarray(rows, dtype=np.float32)


def build_features(df: pd.DataFrame) -> np.ndarray:
    feats = [encode_smiles_column(df[col]) for col in MOLECULE_COLUMNS]
    return np.concatenate(feats, axis=1)

# ──────────────────────────────────────
def get_models():
    """返回 (name, model_constructor) 列表"""
    models = [
        (
            "RandomForest",
            lambda: RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_split=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "XGBoost",
            lambda: xgb.XGBRegressor(
                base_score=0,
                eval_metric="rmse",
                n_estimators=2000,
                early_stopping_rounds=100,
                max_depth=8,
                subsample=0.7,
                learning_rate=0.01,
                random_state=42,
                tree_method="auto",
            ),
        ),
        (
            "CatBoost",
            lambda: cb.CatBoostRegressor(
                objective="RMSE",
                eval_metric="RMSE",
                iterations=5000,
                bagging_temperature=0.5,
                colsample_bylevel=0.7,
                learning_rate=0.02,
                od_wait=25,
                max_depth=7,
                l2_leaf_reg=1.5,
                min_data_in_leaf=1000,
                random_strength=0.65,
                verbose=True,
                metric_period=100,
                use_best_model=True,
            ),
        ),
        (
            "LightGBM",
            lambda: lgb.LGBMRegressor(
                objective="mse",
                n_estimators=5000,
                num_leaves=256,
                subsample=0.6,
                colsample_bytree=0.6,
                learning_rate=0.00871,
                n_jobs=4,
                verbosity=-1,
                importance_type="gain",
            ),
        ),
    ]
    return models


# ──────────────────────────────────────
def evaluate_by_rxntype(
    rxn_df: pd.DataFrame,
    model_name: str,
    model_fn,
    rxn_type_value: int,
    output_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[dict], dict]:
    start_time = perf_counter()

    X = build_features(rxn_df)
    y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split_indices = list(kf.split(X))

    fold_iter = enumerate(split_indices, start=1)
    if HAS_TQDM:
        fold_iter = enumerate(
            tqdm(
                split_indices,
                total=n_splits,
                desc=f"rxn_{rxn_type_value} | {model_name}",
                leave=False,
            ),
            start=1,
        )

    rxn_dir = output_dir / f"rxn_{rxn_type_value}"
    rxn_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for fold, (train_idx, val_idx) in fold_iter:
        fold_start = perf_counter()

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_fn()

       
        if model_name == "LightGBM":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.callback.early_stopping(stopping_rounds=100),
                    lgb.callback.log_evaluation(period=0),
                ],
            )
        elif model_name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif model_name == "CatBoost":
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                verbose=0,
            )
        else:  # RandomForest
            model.fit(X_train, y_train)

        preds = model.predict(X_val)

        r2 = float(r2_score(y_val, preds))
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))

        best_iter = getattr(
            model, "best_iteration_", getattr(model, "best_iteration", -1)
        )

      
        if model_name == "LightGBM":
            model.booster_.save_model(str(rxn_dir / f"{model_name}_fold{fold}.txt"))
        elif model_name == "CatBoost":
            model.save_model(str(rxn_dir / f"{model_name}_fold{fold}.cbm"))
        else:
            import pickle

            with open(rxn_dir / f"{model_name}_fold{fold}.pkl", "wb") as f:
                pickle.dump(model, f)

        fold_rows.append(
            {
                "model": model_name,
                "rxntype": int(rxn_type_value),
                "fold": int(fold),
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "best_iteration": int(best_iter) if best_iter != -1 else None,
                "fold_seconds": float(perf_counter() - fold_start),
            }
        )

    r2_values = np.array([x["r2"] for x in fold_rows], dtype=float)
    rmse_values = np.array([x["rmse"] for x in fold_rows], dtype=float)
    mae_values = np.array([x["mae"] for x in fold_rows], dtype=float)
    sec_values = np.array([x["fold_seconds"] for x in fold_rows], dtype=float)

    summary_row = {
        "model": model_name,
        "rxntype": int(rxn_type_value),
        "n_samples": int(len(rxn_df)),
        "r2_mean": float(np.mean(r2_values)),
        "r2_sd": float(np.std(r2_values, ddof=1)),
        "r2_mean_pm_sd": f"{np.mean(r2_values):.6f} ± {np.std(r2_values, ddof=1):.6f}",
        "rmse_mean": float(np.mean(rmse_values)),
        "rmse_sd": float(np.std(rmse_values, ddof=1)),
        "mae_mean": float(np.mean(mae_values)),
        "mae_sd": float(np.std(mae_values, ddof=1)),
        "fold_seconds_mean": float(np.mean(sec_values)),
        "fold_seconds_sd": float(np.std(sec_values, ddof=1)),
        "total_seconds": float(perf_counter() - start_time),
    }
    return fold_rows, summary_row


# ──────────────────────────────────────
def load_train_data(dataset_dir: Path) -> pd.DataFrame:
    round1_path = dataset_dir / "round1_train_data.csv"
    round2_path = dataset_dir / "round2_train_data.csv"

    if not round1_path.exists():
        raise FileNotFoundError(f"找不到文件: {round1_path}")
    if not round2_path.exists():
        raise FileNotFoundError(f"找不到文件: {round2_path}")

    df1 = pd.read_csv(round1_path).copy()
    df2 = pd.read_csv(round2_path).copy()

    if "rxntype" not in df1.columns:
        df1["rxntype"] = 1
    if "rxntype" not in df2.columns:
        df2["rxntype"] = 2

    df = pd.concat([df1, df2], axis=0, ignore_index=True)

    required_cols = MOLECULE_COLUMNS + [TARGET_COLUMN, "rxntype"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"训练集缺少必要列: {missing}")

    df["rxntype"] = pd.to_numeric(df["rxntype"], errors="coerce")
    if df["rxntype"].isna().any():
        raise ValueError("rxntype 列包含无法解析为数字的值")
    df["rxntype"] = df["rxntype"].astype(int)

    return df



# ──────────────────────────────────────
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent.parent / "data"
    output_root = script_dir.parent / "ckpt-avalon-fp-comparison"
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_train_data(dataset_dir)
    rxn_groups = {int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")}

    print(f"数据总量: {len(df)}, 反应类型数: {len(rxn_groups)}")
    for rt in sorted(rxn_groups):
        print(f"  rxntype={rt}: {len(rxn_groups[rt])} 样本")

    all_start = perf_counter()
    models = get_models()

    all_fold_records = []
    all_summary_records = []

    for model_name, model_fn in models:
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"{'='*60}")

        model_output_dir = output_root / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        fold_records = []
        summary_records = []

        for rxntype in sorted(rxn_groups.keys()):
            rxn_df = rxn_groups[rxntype]
            print(f"\n  rxntype={rxntype}, n={len(rxn_df)}")

            try:
                fold_rows, summary_row = evaluate_by_rxntype(
                    rxn_df=rxn_df,
                    model_name=model_name,
                    model_fn=model_fn,
                    rxn_type_value=rxntype,
                    output_dir=model_output_dir,
                    n_splits=5,
                    random_state=42,
                )
                fold_records.extend(fold_rows)
                summary_records.append(summary_row)

                print(
                    f"  rxntype={rxntype} | R2: {summary_row['r2_mean_pm_sd']} | "
                    f"RMSE: {summary_row['rmse_mean']:.6f} | "
                    f"MAE: {summary_row['mae_mean']:.6f}"
                )
            except Exception as exc:
                print(f"  [跳过] {model_name} | rxntype={rxntype} 失败: {exc}")

       
        fold_df = pd.DataFrame(fold_records)
        summary_df = pd.DataFrame(summary_records)

        fold_df.to_csv(model_output_dir / "cv_fold_metrics.csv", index=False)
        summary_df.to_csv(model_output_dir / "cv_summary_metrics.csv", index=False)
        print(f"\n  已保存: {model_output_dir}")

        all_fold_records.extend(fold_records)
        all_summary_records.extend(summary_records)

   
    all_fold_df = pd.DataFrame(all_fold_records)
    all_summary_df = pd.DataFrame(all_summary_records)
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_fold_df.to_csv(results_dir / "avalon_all_models_fold_metrics.csv", index=False)
    all_summary_df.to_csv(results_dir / "avalon_all_models_summary_metrics.csv", index=False)

  
    print(f"\n{'='*80}")
    print("模型对比排名 (按 R2 降序)")
    print(f"{'='*80}")
    print(
        f"{'排名':<4} {'模型':<16} {'rxntype':<10} {'样本数':<10} "
        f"{'R2 (mean±sd)':<34} {'RMSE':<12} {'MAE':<12}"
    )
    print("-" * 100)
    ranked = all_summary_df.sort_values("r2_mean", ascending=False).reset_index(
        drop=True
    )
    for i, row in ranked.iterrows():
        print(
            f"{i+1:<4} {row['model']:<16} {row['rxntype']:<10} {row['n_samples']:<10} "
            f"{row['r2_mean_pm_sd']:<34} {row['rmse_mean']:<12.6f} {row['mae_mean']:<12.6f}"
        )

   
    print(f"\n{'='*80}")
    print("各 rxntype 最佳模型")
    print(f"{'='*80}")
    for rt in sorted(all_summary_df["rxntype"].unique()):
        sub = all_summary_df[all_summary_df["rxntype"] == rt]
        best = sub.loc[sub["r2_mean"].idxmax()]
        print(
            f"  rxntype={rt}: {best['model']} "
            f"(R2={best['r2_mean']:.6f}, RMSE={best['rmse_mean']:.6f})"
        )

   
    print(f"\n{'='*80}")
    print("各模型所有 rxntype 平均 R2")
    print(f"{'='*80}")
    model_avg = all_summary_df.groupby("model")["r2_mean"].mean().sort_values(
        ascending=False
    )
    for model_name, avg_r2 in model_avg.items():
        print(f"  {model_name}: R2={avg_r2:.6f}")

    print(f"\n模型保存目录: {output_root}")
    print(f"结果保存目录: {results_dir}")
    print(f"总耗时: {perf_counter() - all_start:.2f}s")


if __name__ == "__main__":
    main()
