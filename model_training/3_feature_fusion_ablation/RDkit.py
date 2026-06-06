# LightGBM: Pure RDKit Descriptor Baseline
# Classified by rxntype, 5-fold cross-validation
# Only uses extra-rdkit features, no concatenation of any fingerprints
import gzip
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

TARGET_COLUMN = "Yield"


# ──────────────────────────────────────
# RDKit 描述符加载（从 gz 文件）
# ──────────────────────────────────────
def load_rdkit_features(rdkit_dir: Path, rxntype: int) -> np.ndarray:
    gz_path = rdkit_dir / f"train-rdkitfeature-rxn{rxntype}.gz"
    if not gz_path.exists():
        raise FileNotFoundError(f"找不到 RDKit 特征文件: {gz_path}")

    with gzip.open(gz_path, "rb") as f:
        raw = f.read().decode()

    data = []
    for line in raw.strip().split("\n"):
        vals = [float(x) for x in line.split(",")]
        data.append(vals)

    arr = np.asarray(data, dtype=np.float32)

    # 处理 NaN / Inf
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    problem_mask = nan_mask | inf_mask
    if problem_mask.any():
        col_means = np.nanmean(arr, axis=0)
        col_means[np.isnan(col_means)] = 0.0
        for j in range(arr.shape[1]):
            mask = problem_mask[:, j]
            if mask.any():
                arr[mask, j] = col_means[j]

    return arr


# ──────────────────────────────────────
# 数据加载
# ──────────────────────────────────────
MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]


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
# LightGBM 参数（与融合实验一致）
# ──────────────────────────────────────
LGB_PARAMS = {
    "objective": "mse",
    "n_estimators": 5000,
    "num_leaves": 256,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "learning_rate": 0.00871,
    "n_jobs": 4,
    "verbosity": -1,
    "importance_type": "gain",
}


# ──────────────────────────────────────
# 训练与评估：纯 RDKit 描述符
# ──────────────────────────────────────
def evaluate_rdkit(
    rdkit_feats: np.ndarray,
    y: np.ndarray,
    rxn_type_value: int,
    output_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[dict], dict]:
    start_time = perf_counter()

    X = rdkit_feats
    print(f"    特征维度: RDKit={X.shape[1]}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split_indices = list(kf.split(X))

    fold_iter = enumerate(split_indices, start=1)
    if HAS_TQDM:
        fold_iter = enumerate(
            tqdm(split_indices, total=n_splits, desc=f"rxn_{rxn_type_value} rdkit", leave=False),
            start=1,
        )

    rxn_dir = output_dir / f"rxn_{rxn_type_value}"
    rxn_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for fold, (train_idx, val_idx) in fold_iter:
        fold_start = perf_counter()

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=0),
            ],
        )

        preds = model.predict(X_val)
        r2 = float(r2_score(y_val, preds))
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))

        model.booster_.save_model(str(rxn_dir / f"lgbm_rdkit_fold{fold}.txt"))

        fold_rows.append(
            {
                "rxntype": int(rxn_type_value),
                "fold": int(fold),
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "best_iteration": int(model.best_iteration_),
                "fold_seconds": float(perf_counter() - fold_start),
            }
        )

    r2_vals = np.array([x["r2"] for x in fold_rows])
    rmse_vals = np.array([x["rmse"] for x in fold_rows])
    mae_vals = np.array([x["mae"] for x in fold_rows])
    sec_vals = np.array([x["fold_seconds"] for x in fold_rows])

    summary = {
        "rxntype": int(rxn_type_value),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "r2_mean": float(np.mean(r2_vals)),
        "r2_sd": float(np.std(r2_vals, ddof=1)),
        "r2_mean_pm_sd": f"{np.mean(r2_vals):.6f} ± {np.std(r2_vals, ddof=1):.6f}",
        "rmse_mean": float(np.mean(rmse_vals)),
        "rmse_sd": float(np.std(rmse_vals, ddof=1)),
        "mae_mean": float(np.mean(mae_vals)),
        "mae_sd": float(np.std(mae_vals, ddof=1)),
        "fold_seconds_mean": float(np.mean(sec_vals)),
        "total_seconds": float(perf_counter() - start_time),
    }
    return fold_rows, summary


# ──────────────────────────────────────
# 特征重要性分析
# ──────────────────────────────────────
def analyze_importance(
    rdkit_feats: np.ndarray,
    y: np.ndarray,
    rxn_type_value: int,
    output_dir: Path,
    top_k: int = 30,
) -> None:
    X = rdkit_feats

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(kf.split(X))

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X[train_idx],
        y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        callbacks=[
            lgb.callback.early_stopping(stopping_rounds=100),
            lgb.callback.log_evaluation(period=0),
        ],
    )

    importances = model.feature_importances_

    # 构建特征名: 5 个分子列 x 210 描述符
    N_DESC = 210
    feature_names = []
    for col in MOLECULE_COLUMNS:
        feature_names.extend([f"RDKit_{col}_{i}" for i in range(N_DESC)])
    components = [col for col in MOLECULE_COLUMNS for _ in range(N_DESC)]

    imp_df = (
        pd.DataFrame(
            {
                "feature": feature_names[: len(importances)],
                "importance": importances,
                "component": components[: len(importances)],
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    rxn_dir = output_dir / f"rxn_{rxn_type_value}"
    rxn_dir.mkdir(parents=True, exist_ok=True)
    imp_df.head(top_k).to_csv(rxn_dir / "top_features.csv", index=False)

    comp_imp = imp_df.groupby("component")["importance"].sum()
    total_imp = comp_imp.sum()
    print("    各分子列重要性占比:")
    for col in MOLECULE_COLUMNS:
        pct = comp_imp.get(col, 0) / total_imp * 100
        print(f"      {col}: {pct:.1f}%")

    top10 = imp_df.head(10)
    for _, row in top10.iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")


# ──────────────────────────────────────
# 主流程
# ──────────────────────────────────────
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent.parent / "data"
    rdkit_dir = dataset_dir / "extra-rdkit"
    output_root = script_dir.parent / "ckpt-rdkit-baseline"
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_train_data(dataset_dir)
    rxn_groups = {int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")}

    print(f"数据总量: {len(df)}, 反应类型数: {len(rxn_groups)}")
    for rt in sorted(rxn_groups):
        print(f"  rxntype={rt}: {len(rxn_groups[rt])} 样本")

    # 校验 RDKit 特征文件
    print("\n校验 RDKit 特征文件...")
    for rt in sorted(rxn_groups):
        gz_path = rdkit_dir / f"train-rdkitfeature-rxn{rt}.gz"
        if not gz_path.exists():
            print(f"  [警告] rxntype={rt}: 缺少 {gz_path}")
            continue
        rdkit_arr = load_rdkit_features(rdkit_dir, rt)
        expected = len(rxn_groups[rt])
        actual = rdkit_arr.shape[0]
        status = "OK" if actual == expected else "MISMATCH"
        print(
            f"  rxntype={rt}: 期望 {expected}, 实际 {actual}, 特征数 {rdkit_arr.shape[1]} [{status}]"
        )

    all_start = perf_counter()

    all_folds = []
    all_summary = []

    print(f"\n{'=' * 70}")
    print("纯 RDKit 描述符基线训练")
    print(f"{'=' * 70}")

    for rxntype in sorted(rxn_groups.keys()):
        rxn_df = rxn_groups[rxntype]
        print(f"\n  rxntype={rxntype}, n={len(rxn_df)}")

        try:
            rdkit_feats = load_rdkit_features(rdkit_dir, rxntype)
        except FileNotFoundError as e:
            print(f"  [跳过] {e}")
            continue

        if rdkit_feats.shape[0] != len(rxn_df):
            print(f"  [跳过] 样本数不匹配: data={len(rxn_df)}, rdkit={rdkit_feats.shape[0]}")
            continue

        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        fold_rows, summary_row = evaluate_rdkit(
            rdkit_feats=rdkit_feats,
            y=y,
            rxn_type_value=rxntype,
            output_dir=output_root,
            n_splits=5,
            random_state=42,
        )
        all_folds.extend(fold_rows)
        all_summary.append(summary_row)

        print(
            f"  rxntype={rxntype} | R2: {summary_row['r2_mean_pm_sd']}"
            f" | RMSE: {summary_row['rmse_mean']:.6f}"
            f" | MAE: {summary_row['mae_mean']:.6f}"
        )

        # 特征重要性分析
        try:
            analyze_importance(
                rdkit_feats=rdkit_feats,
                y=y,
                rxn_type_value=rxntype,
                output_dir=output_root,
            )
        except Exception as e:
            print(f"    [警告] 特征重要性分析失败: {e}")

    # ── 保存结果 ──
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_folds).to_csv(results_dir / "rdkit_baseline_fold_metrics.csv", index=False)
    pd.DataFrame(all_summary).to_csv(results_dir / "rdkit_baseline_summary.csv", index=False)

    # ── 汇总表 ──
    print(f"\n{'=' * 80}")
    print("纯 RDKit 描述符基线 — 汇总")
    print(f"{'=' * 80}")
    print(
        f"{'rxntype':<10} {'n':<8} {'features':<10} "
        f"{'R2':<20} {'RMSE':<14} {'MAE':<14} {'time(s)':<10}"
    )
    print("-" * 86)

    for s in all_summary:
        print(
            f"{s['rxntype']:<10} {s['n_samples']:<8} {s['n_features']:<10} "
            f"{s['r2_mean_pm_sd']:<20} {s['rmse_mean']:<14.6f} {s['mae_mean']:<14.6f} "
            f"{s['total_seconds']:<10.2f}"
        )

    if all_summary:
        avg_r2 = np.mean([s["r2_mean"] for s in all_summary])
        avg_rmse = np.mean([s["rmse_mean"] for s in all_summary])
        avg_mae = np.mean([s["mae_mean"] for s in all_summary])
        print("-" * 86)
        print(
            f"{'平均':<10} {'':8} {'':10} "
            f"{avg_r2:<20.6f} {avg_rmse:<14.6f} {avg_mae:<14.6f}"
        )

    print(f"\n模型保存目录: {output_root}")
    print(f"结果保存目录: {results_dir}")
    print(f"总耗时: {perf_counter() - all_start:.2f}s")


if __name__ == "__main__":
    main()
