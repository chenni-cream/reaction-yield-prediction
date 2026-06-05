# Avalon FP + Top-K RDKit 特征训练 & 模型保存
# 基于消融实验结论: Top-500 RDKit 特征为全局最优配置
# 对每个 rxntype: 获取 importance -> 选 Top-500 RDKit -> 5 折训练 -> 保存模型
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
TOP_K = 500
N_FOLDS = 5

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
    return np.concatenate(
        [encode_smiles_column(df[col]) for col in MOLECULE_COLUMNS], axis=1
    )


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
# 获取 Top-K RDKit 特征列索引
# ──────────────────────────────────────
def get_topk_rdkit_indices(X_full, y, top_k=TOP_K):
    """训练一折获取 importance，返回 Top-K RDKit 特征在 X_full 中的列索引。"""
    feature_names = []
    for col in MOLECULE_COLUMNS:
        feature_names.extend([f"Avalon_{col}_{i}" for i in range(FP_SIZE)])
    for col in MOLECULE_COLUMNS:
        feature_names.extend([f"RDKit_{col}_{i}" for i in range(N_RDKIT)])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(kf.split(X_full))
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_full[train_idx],
        y[train_idx],
        eval_set=[(X_full[val_idx], y[val_idx])],
        callbacks=[
            lgb.callback.early_stopping(stopping_rounds=100),
            lgb.callback.log_evaluation(period=0),
        ],
    )

    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    # 只取 RDKit 特征
    rdkit_imp = imp_df[imp_df["feature"].str.startswith("RDKit_")]
    rdkit_imp = rdkit_imp[rdkit_imp["importance"] > 0]

    indices = []
    for feat in rdkit_imp["feature"].iloc[:top_k]:
        parts = feat.split("_", 2)
        comp = parts[1]
        local_idx = int(parts[2])
        comp_idx = MOLECULE_COLUMNS.index(comp)
        global_idx = len(MOLECULE_COLUMNS) * FP_SIZE + comp_idx * N_RDKIT + local_idx
        indices.append(global_idx)

    return np.array(indices), imp_df


# ──────────────────────────────────────
# 5 折训练 & 保存模型
# ──────────────────────────────────────
def train_and_save(X, y, rxntype, output_dir, n_splits=N_FOLDS, random_state=42):
    rxn_dir = output_dir / f"rxn_{rxntype}"
    rxn_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        fold_start = perf_counter()

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

        preds = model.predict(X[val_idx])
        r2 = float(r2_score(y[val_idx], preds))
        rmse = float(np.sqrt(mean_squared_error(y[val_idx], preds)))
        mae = float(mean_absolute_error(y[val_idx], preds))

        # 保存模型
        model.booster_.save_model(str(rxn_dir / f"lgbm_top{TOP_K}_fold{fold}.txt"))

        fold_rows.append(
            {
                "rxntype": rxntype,
                "fold": fold,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "best_iteration": int(model.best_iteration_),
                "seconds": round(perf_counter() - fold_start, 2),
            }
        )
        print(
            f"    fold {fold}: R2={r2:.6f} RMSE={rmse:.6f} MAE={mae:.6f} "
            f"iter={model.best_iteration_} ({fold_rows[-1]['seconds']}s)"
        )

    return fold_rows


# ──────────────────────────────────────
# 主流程
# ──────────────────────────────────────
def main():
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent.parent / "data"
    rdkit_dir = dataset_dir / "extra-rdkit"
    output_root = script_dir.parent / "ckpt-avalon-rdkit-topk"
    output_root.mkdir(parents=True, exist_ok=True)
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_train_data(dataset_dir)
    rxn_groups = {
        int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")
    }

    print(f"数据总量: {len(df)}, 反应类型数: {len(rxn_groups)}")
    print(f"配置: Avalon FP (10240) + Top-{TOP_K} RDKit")

    all_start = perf_counter()
    all_folds = []

    for rxntype in sorted(rxn_groups.keys()):
        rxn_df = rxn_groups[rxntype]
        print(f"\n{'='*60}")
        print(f"rxntype={rxntype}, n={len(rxn_df)}")

        try:
            rdkit_arr = load_rdkit_features(rdkit_dir, rxntype)
        except FileNotFoundError as e:
            print(f"  [跳过] {e}")
            continue
        if rdkit_arr.shape[0] != len(rxn_df):
            print(
                f"  [跳过] 样本数不匹配: data={len(rxn_df)}, rdkit={rdkit_arr.shape[0]}"
            )
            continue

        # 构建全量特征
        avalon_feats = build_avalon_features(rxn_df)
        X_full = np.concatenate([avalon_feats, rdkit_arr], axis=1)
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        print(f"  全量特征: {X_full.shape[1]} 维")

        # 获取 Top-K RDKit 特征索引
        topk_indices, imp_df = get_topk_rdkit_indices(X_full, y, top_k=TOP_K)
        n_avalon = len(MOLECULE_COLUMNS) * FP_SIZE

        # 保存特征重要性
        imp_dir = output_root / f"rxn_{rxntype}"
        imp_dir.mkdir(parents=True, exist_ok=True)
        imp_df.to_csv(imp_dir / "feature_importance.csv", index=False)

        # 拼接: Avalon 全部 + Top-K RDKit
        selected = np.concatenate([np.arange(n_avalon), topk_indices])
        selected.sort()
        X_selected = X_full[:, selected]

        actual_rdkit = len(topk_indices)
        print(
            f"  选中: Avalon {n_avalon} + RDKit Top-{actual_rdkit} = {X_selected.shape[1]} 维"
        )

        # 5 折训练并保存模型
        fold_rows = train_and_save(X_selected, y, rxntype, output_root)
        all_folds.extend(fold_rows)

        r2_vals = [f["r2"] for f in fold_rows]
        print(
            f"  >>> 平均 R2={np.mean(r2_vals):.6f} ± {np.std(r2_vals, ddof=1):.6f}"
        )

    # ── 汇总 ──
    if not all_folds:
        print("无训练结果")
        return

    folds_df = pd.DataFrame(all_folds)

    # 按 rxntype 汇总
    summary_rows = []
    for rt in sorted(folds_df["rxntype"].unique()):
        sub = folds_df[folds_df["rxntype"] == rt]
        summary_rows.append(
            {
                "rxntype": rt,
                "n_folds": len(sub),
                "r2_mean": sub["r2"].mean(),
                "r2_sd": sub["r2"].std(ddof=1),
                "rmse_mean": sub["rmse"].mean(),
                "mae_mean": sub["mae"].mean(),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    print(f"\n{'='*70}")
    print(f"汇总: Avalon FP + Top-{TOP_K} RDKit 融合模型")
    print(f"{'='*70}")
    print(
        f"{'rxntype':<10} {'R2_mean':<14} {'R2_sd':<14} {'RMSE':<14} {'MAE':<14}"
    )
    print("-" * 66)
    for _, row in summary_df.iterrows():
        print(
            f"{row['rxntype']:<10} {row['r2_mean']:<14.6f} {row['r2_sd']:<14.6f} "
            f"{row['rmse_mean']:<14.6f} {row['mae_mean']:<14.6f}"
        )

    avg_r2 = summary_df["r2_mean"].mean()
    print(f"\n  全局平均 R2: {avg_r2:.6f}")

    # 保存结果
    folds_df.to_csv(
        results_dir / f"avalon_rdkit_top{TOP_K}_fold_metrics.csv", index=False
    )
    summary_df.to_csv(
        results_dir / f"avalon_rdkit_top{TOP_K}_summary.csv", index=False
    )

    print(f"\n模型保存目录: {output_root}")
    print(f"  每折模型: ckpt-avalon-rdkit-topk/rxn_{{type}}/lgbm_top{TOP_K}_fold{{i}}.txt")
    print(f"结果保存目录: {results_dir}")
    print(f"总耗时: {perf_counter() - all_start:.1f}s")


if __name__ == "__main__":
    main()
