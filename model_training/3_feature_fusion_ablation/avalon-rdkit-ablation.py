# Avalon + RDKit Ablation Experiment
# Full Avalon+RDKit training, then select the Top-K RDKit feature according to LightGBM importance
# Classify by rxntype, conduct a 5 cross-validation, and find the optimal K value for each rxntype
import gzip
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    from rdkit.Avalon import pyAvalonTools
except ImportError:
    pyAvalonTools = None

RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048
N_RDKIT = 210

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
# Avalon 指纹
# ──────────────────────────────────────
def bitvect_to_numpy(bitvect, n_bits):
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def avalon_fp(mol, n_bits):
    fp = pyAvalonTools.GetAvalonFP(mol, n_bits)
    return bitvect_to_numpy(fp, n_bits)


def encode_smiles_column(smiles_series):
    smiles = smiles_series.fillna("").astype(str).tolist()
    rows = []
    zero = np.zeros((FP_SIZE,), dtype=np.uint8)
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(zero if mol is None else avalon_fp(mol, FP_SIZE))
    return np.asarray(rows, dtype=np.float32)


def build_avalon_features(df):
    return np.concatenate([encode_smiles_column(df[col]) for col in MOLECULE_COLUMNS], axis=1)


# ──────────────────────────────────────
# RDKit 特征加载
# ──────────────────────────────────────
def load_rdkit_features(rdkit_dir, rxntype):
    gz_path = rdkit_dir / f"train-rdkitfeature-rxn{rxntype}.gz"
    if not gz_path.exists():
        raise FileNotFoundError(f"找不到: {gz_path}")

    with gzip.open(gz_path, "rb") as f:
        raw = f.read().decode()

    data = []
    for line in raw.strip().split("\n"):
        data.append([float(x) for x in line.split(",")])

    arr = np.asarray(data, dtype=np.float32)

    problem_mask = np.isnan(arr) | np.isinf(arr)
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
def load_train_data(dataset_dir):
    df1 = pd.read_csv(dataset_dir / "round1_train_data.csv").copy()
    df2 = pd.read_csv(dataset_dir / "round2_train_data.csv").copy()
    if "rxntype" not in df1.columns:
        df1["rxntype"] = 1
    if "rxntype" not in df2.columns:
        df2["rxntype"] = 2
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    df["rxntype"] = pd.to_numeric(df["rxntype"], errors="coerce").astype(int)
    return df


# ──────────────────────────────────────
# 训练评估
# ──────────────────────────────────────
def train_and_evaluate(X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=0),
            ],
        )
        preds = model.predict(X[val_idx])
        fold_rows.append({
            "fold": fold,
            "r2": float(r2_score(y[val_idx], preds)),
            "rmse": float(np.sqrt(mean_squared_error(y[val_idx], preds))),
            "mae": float(mean_absolute_error(y[val_idx], preds)),
        })

    r2_vals = np.array([x["r2"] for x in fold_rows])
    rmse_vals = np.array([x["rmse"] for x in fold_rows])
    mae_vals = np.array([x["mae"] for x in fold_rows])
    return {
        "r2_mean": float(np.mean(r2_vals)),
        "r2_sd": float(np.std(r2_vals, ddof=1)),
        "rmse_mean": float(np.mean(rmse_vals)),
        "mae_mean": float(np.mean(mae_vals)),
    }


def get_feature_importance(X, y, feature_names):
    """训练一折获取 feature importance"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(kf.split(X))
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X[train_idx], y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        callbacks=[
            lgb.callback.early_stopping(stopping_rounds=100),
            lgb.callback.log_evaluation(period=0),
        ],
    )
    return pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────
# 主流程
# ──────────────────────────────────────
def main():
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent.parent / "data"
    rdkit_dir = dataset_dir / "extra-rdkit"
    output_root = script_dir.parent / "ckpt-ablation"
    output_root.mkdir(parents=True, exist_ok=True)
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_train_data(dataset_dir)
    rxn_groups = {int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")}

    print(f"数据总量: {len(df)}, 反应类型数: {len(rxn_groups)}")

    # Top-K 候选值
    k_values = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1050]
    all_results = []

    all_start = perf_counter()

    # ── Step 1: 全量 Avalon+RDKit ──
    print(f"\n{'='*70}")
    print("Step 1: Avalon + RDKit 全量 (11290 维)")
    print(f"{'='*70}")

    for rxntype in sorted(rxn_groups.keys()):
        rxn_df = rxn_groups[rxntype]
        try:
            rdkit_arr = load_rdkit_features(rdkit_dir, rxntype)
        except FileNotFoundError:
            continue
        if rdkit_arr.shape[0] != len(rxn_df):
            continue

        avalon_feats = build_avalon_features(rxn_df)
        X_full = np.concatenate([avalon_feats, rdkit_arr], axis=1)
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        summary = train_and_evaluate(X_full, y)
        all_results.append({
            "config": "Avalon+RDKit_all",
            "rxntype": rxntype,
            "n_samples": len(rxn_df),
            "n_features": X_full.shape[1],
            **summary,
        })
        print(f"  rxntype={rxntype} | R2={summary['r2_mean']:.6f} | 特征数={X_full.shape[1]}")

    # ── Step 2: Top-K RDKit 消融 ──
    print(f"\n{'='*70}")
    print("Step 2: Avalon + Top-K RDKit 消融")
    print(f"{'='*70}")

    for rxntype in sorted(rxn_groups.keys()):
        rxn_df = rxn_groups[rxntype]
        print(f"\n  rxntype={rxntype}, n={len(rxn_df)}")

        try:
            rdkit_arr = load_rdkit_features(rdkit_dir, rxntype)
        except FileNotFoundError:
            continue
        if rdkit_arr.shape[0] != len(rxn_df):
            continue

        avalon_feats = build_avalon_features(rxn_df)
        X_full = np.concatenate([avalon_feats, rdkit_arr], axis=1)
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        # 构建特征名
        feature_names = []
        for col in MOLECULE_COLUMNS:
            feature_names.extend([f"Avalon_{col}_{i}" for i in range(FP_SIZE)])
        for col in MOLECULE_COLUMNS:
            feature_names.extend([f"RDKit_{col}_{i}" for i in range(N_RDKIT)])

        # 获取 importance
        imp_df = get_feature_importance(X_full, y, feature_names)

        imp_output = output_root / f"rxn_{rxntype}"
        imp_output.mkdir(parents=True, exist_ok=True)
        imp_df.to_csv(imp_output / "feature_importance.csv", index=False)

        # 提取 RDKit 特征 importance 排名
        rdkit_imp = imp_df[imp_df["feature"].str.startswith("RDKit_")].copy()
        n_nonzero = (rdkit_imp["importance"] > 0).sum()
        print(f"    RDKit 非零 importance 特征数: {n_nonzero}")

        # 解析 RDKit 特征名 -> 全局列索引
        rdkit_global_indices = []
        for _, row in rdkit_imp.iterrows():
            feat = row["feature"]
            parts = feat.split("_", 2)
            comp = parts[1]
            local_idx = int(parts[2])
            comp_idx = MOLECULE_COLUMNS.index(comp)
            global_idx = len(MOLECULE_COLUMNS) * FP_SIZE + comp_idx * N_RDKIT + local_idx
            rdkit_global_indices.append(global_idx)

        n_avalon = len(MOLECULE_COLUMNS) * FP_SIZE

        # 对每个 K 值做消融
        rxn_start = perf_counter()
        for k in k_values:
            if k > n_nonzero:
                break

            selected_indices = rdkit_global_indices[:k]
            all_indices = np.concatenate([
                np.arange(n_avalon),       # Avalon 全部保留
                np.array(selected_indices), # Top-K RDKit
            ])
            X_topk = X_full[:, all_indices]

            summary = train_and_evaluate(X_topk, y)
            all_results.append({
                "config": f"Avalon+RDKit_Top{k}",
                "rxntype": rxntype,
                "n_samples": len(rxn_df),
                "n_features": X_topk.shape[1],
                **summary,
            })

        print(f"    完成 {k} 个 K 值, 耗时 {perf_counter() - rxn_start:.1f}s")

    # ── 汇总 ──
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / "ablation_topk_results.csv", index=False)

    # 核心目标: 找到一个全局统一 K，使 8 个 rxntype 平均 R2 最优
    print(f"\n{'='*80}")
    print("全局统一 K 值排名 (8 个 rxntype 加权平均 R2)")
    print(f"{'='*80}")

    config_avg = results_df.groupby("config").agg(
        simple_r2=("r2_mean", "mean"),
        weighted_r2=("r2_mean", lambda g: np.average(g, weights=results_df.loc[g.index, "n_samples"])),
        simple_rmse=("rmse_mean", "mean"),
        weighted_rmse=("rmse_mean", lambda g: np.average(g, weights=results_df.loc[g.index, "n_samples"])),
        n_features=("n_features", "first"),
    ).sort_values("simple_r2", ascending=False)

    print(f"{'排名':<4} {'配置':<30} {'简单平均 R2':<14} {'加权平均 R2':<14} {'简单 RMSE':<14} {'特征数':<10}")
    print("-" * 96)
    for i, (config, row) in enumerate(config_avg.iterrows(), 1):
        print(f"{i:<4} {config:<30} {row['simple_r2']:<14.6f} {row['weighted_r2']:<14.6f} "
              f"{row['simple_rmse']:<14.6f} {row['n_features']:<10.0f}")

    # 最佳全局 K (按简单平均 R2 排序，体现通用性)
    best_config = config_avg.index[0]
    best_r2 = config_avg.iloc[0]["simple_r2"]
    print(f"\n>>> 全局最优配置: {best_config}")
    print(f">>> 简单平均 R2: {best_r2:.6f} (通用性主指标)")
    print(f">>> 加权平均 R2: {config_avg.iloc[0]['weighted_r2']:.6f}")

    # 展示该配置下各 rxntype 的表现
    print(f"\n{'='*80}")
    print(f"全局最优配置 [{best_config}] 各 rxntype 表现")
    print(f"{'='*80}")
    best_sub = results_df[results_df["config"] == best_config]
    print(f"{'rxntype':<10} {'n':<10} {'R2':<14} {'RMSE':<14} {'MAE':<14}")
    print("-" * 62)
    for _, row in best_sub.iterrows():
        print(f"{row['rxntype']:<10} {row['n_samples']:<10.0f} {row['r2_mean']:<14.6f} "
              f"{row['rmse_mean']:<14.6f} {row['mae_mean']:<14.6f}")

    # 保存全局最优配置
    config_avg.to_csv(results_dir / "ablation_global_k_ranking.csv")

    best_df = best_sub[["rxntype", "n_samples", "r2_mean", "r2_sd", "rmse_mean", "mae_mean", "n_features"]].copy()
    best_df.to_csv(results_dir / "ablation_best_global_k_results.csv", index=False)

    print(f"\n结果保存: {results_dir / 'ablation_topk_results.csv'}")
    print(f"排名表:   {results_dir / 'ablation_global_k_ranking.csv'}")
    print(f"最优结果: {results_dir / 'ablation_best_global_k_results.csv'}")
    print(f"总耗时: {perf_counter() - all_start:.1f}s")


if __name__ == "__main__":
    main()
