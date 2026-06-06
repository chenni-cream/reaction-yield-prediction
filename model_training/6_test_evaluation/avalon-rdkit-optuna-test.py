"""
avalon-rdkit-optuna-test.py
The generalized validation of the Avalon + Top-500 RDKit fusion model tuned using Optuna was conducted on the round1 / round2 test set.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent / "ckpt-optuna" / "Avalon_RDKit"
RDKIT_DIR = SCRIPT_DIR.parent.parent / "data" / "extra-rdkit"


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


def encode_avalon_column(smiles_series):
    smiles = smiles_series.fillna("").astype(str).tolist()
    rows = []
    zero = np.zeros((FP_SIZE,), dtype=np.uint8)
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(zero if mol is None else avalon_fp(mol, FP_SIZE))
    return np.asarray(rows, dtype=np.float32)


def build_avalon_features(df):
    return np.concatenate(
        [encode_avalon_column(df[col]) for col in MOLECULE_COLUMNS], axis=1
    )


# ──────────────────────────────────────
# RDKit 描述符特征（查表 + 实时计算）
# ──────────────────────────────────────
_GZ_ORDER_DF = pd.read_csv(RDKIT_DIR / "train-rdkitfeature-Reactant1_feature_names.csv")
_GZ_DESC_NAMES = _GZ_ORDER_DF.iloc[:, 0].tolist()
N_RDKIT_DESC = len(_GZ_DESC_NAMES)  # 210

_JSON_PRIORITY = {
    "Reactant1": ["test-rdkitfeature-Reactant1.json",
                   "train-rdkitfeature-Reactant1.json"],
    "Reactant2": ["test-rdkitfeature-Reactant2.json",
                   "train-rdkitfeature-Reactant2.json"],
    "Product":   ["test-rdkitfeature-Product.json",
                   "train-rdkitfeature-Product.json"],
    "Additive":  ["test-rdkitfeature-Additive.json",
                   "train-rdkitfeature-Additive-nosplit.json",
                   "train-rdkitfeature-Additive.json"],
    "Solvent":   ["test-rdkitfeature-Solvent.json",
                   "train-rdkitfeature-Solvent-nosplit.json",
                   "train-rdkitfeature-Solvent.json"],
}


def _load_json_lookup(filenames):
    for fname in filenames:
        path = RDKIT_DIR / fname
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    return None


def _compute_rdkit_vec(mol):
    from rdkit.Chem import Descriptors as Desc
    vals = Desc.CalcMolDescriptors(mol)
    vec = []
    for name in _GZ_DESC_NAMES:
        v = vals.get(name, 0.0)
        if v is None or not np.isfinite(v):
            v = 0.0
        vec.append(float(v))
    return np.array(vec, dtype=np.float32)


def compute_rdkit_descriptors_for_column(smiles_series, col_name):
    smiles = smiles_series.fillna("").astype(str).tolist()
    zero = np.zeros((N_RDKIT_DESC,), dtype=np.float32)
    lookup = _load_json_lookup(_JSON_PRIORITY.get(col_name, []))

    rows = []
    hit_count = 0
    miss_count = 0

    for smi in smiles:
        if lookup is not None and smi in lookup:
            arr = np.array(lookup[smi][:N_RDKIT_DESC], dtype=np.float32)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            rows.append(arr)
            hit_count += 1
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append(zero)
            else:
                rows.append(_compute_rdkit_vec(mol))
            miss_count += 1

    print(f"    {col_name}: 查表命中 {hit_count}, 实时计算 {miss_count}")
    return np.asarray(rows, dtype=np.float32)


def build_rdkit_features(df):
    feats = [
        compute_rdkit_descriptors_for_column(df[col], col)
        for col in MOLECULE_COLUMNS
    ]
    return np.concatenate(feats, axis=1)


# ──────────────────────────────────────
# 特征选择：从 importance CSV 恢复 Top-K RDKit 列索引
# ──────────────────────────────────────
def get_selected_indices(rxntype):
    """从保存的 feature_importance.csv 中恢复训练时的列选择。"""
    imp_path = MODEL_ROOT / f"rxn_{rxntype}" / "feature_importance.csv"
    if not imp_path.exists():
        raise FileNotFoundError(f"找不到特征重要性文件: {imp_path}")

    imp_df = pd.read_csv(imp_path)

    # 筛选 RDKit 特征，取 importance > 0 的前 Top-K 个
    rdkit_imp = imp_df[imp_df["feature"].str.startswith("RDKit_")].copy()
    rdkit_imp = rdkit_imp[rdkit_imp["importance"] > 0].reset_index(drop=True)

    n_avalon = len(MOLECULE_COLUMNS) * FP_SIZE  # 10240
    indices = []
    for feat in rdkit_imp["feature"].iloc[:TOP_K]:
        parts = feat.split("_", 2)
        comp = parts[1]
        local_idx = int(parts[2])
        comp_idx = MOLECULE_COLUMNS.index(comp)
        global_idx = n_avalon + comp_idx * N_RDKIT + local_idx
        indices.append(global_idx)

    # 拼接: Avalon 全部 + Top-K RDKit
    selected = np.concatenate([np.arange(n_avalon), np.array(indices)])
    selected.sort()
    return selected


# ──────────────────────────────────────
# 模型加载与预测
# ──────────────────────────────────────
def predict_with_ensemble(X, rxntype):
    rxn_dir = MODEL_ROOT / f"rxn_{rxntype}"
    if not rxn_dir.exists():
        raise FileNotFoundError(f"找不到模型目录: {rxn_dir}")

    preds_list = []
    for fold in range(1, N_FOLDS + 1):
        model_path = rxn_dir / f"lgbm_fold{fold}.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        booster = lgb.Booster(model_file=str(model_path))
        preds_list.append(booster.predict(X))

    return np.mean(preds_list, axis=0)


# ──────────────────────────────────────
# 评估逻辑
# ──────────────────────────────────────
def evaluate_dataset(df, dataset_name):
    print(f"\n{'='*70}")
    print(f" 数据集: {dataset_name}  |  样本数: {len(df)}")
    print(f"{'='*70}")

    results = []
    all_y_true = []
    all_preds = []

    for rxntype in sorted(df["rxntype"].unique()):
        sub = df[df["rxntype"] == rxntype].reset_index(drop=True)
        print(f"\n  rxntype={rxntype}, n={len(sub)}")

        avalon = build_avalon_features(sub)
        rdkit = build_rdkit_features(sub)
        X_full = np.concatenate([avalon, rdkit], axis=1)

        selected = get_selected_indices(rxntype)
        X = X_full[:, selected]
        y_true = sub[TARGET_COLUMN].to_numpy(dtype=np.float32)

        print(f"    全量特征: {X_full.shape[1]}, 选中: {X.shape[1]}")

        preds = predict_with_ensemble(X, rxntype)

        all_y_true.append(y_true)
        all_preds.append(preds)

        r2 = r2_score(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)

        results.append({
            "dataset": dataset_name,
            "rxntype": rxntype,
            "n_samples": len(sub),
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
        })
        print(f"    R2: {r2:.6f}  |  RMSE: {rmse:.6f}  |  MAE: {mae:.6f}")

    pooled_y = np.concatenate(all_y_true)
    pooled_p = np.concatenate(all_preds)
    return pd.DataFrame(results), pooled_y, pooled_p


def print_summary(all_results, global_y=None, global_p=None):
    print(f"\n{'='*70}")
    print(" 汇总结果 (Avalon FP + Top-500 RDKit, Optuna 调优)")
    print(f"{'='*70}")
    print(
        f"{'数据集':<12} {'rxntype':<10} {'样本数':<10} "
        f"{'R2':<14} {'RMSE':<14} {'MAE':<14}"
    )
    print("-" * 70)
    for _, row in all_results.iterrows():
        print(
            f"{row['dataset']:<12} {int(row['rxntype']):<10} {int(row['n_samples']):<10} "
            f"{row['r2']:<14.6f} {row['rmse']:<14.6f} {row['mae']:<14.6f}"
        )

    print(f"\n{'='*70}")
    print(" 整体指标（按数据集）")
    print(f"{'='*70}")
    for ds_name in all_results["dataset"].unique():
        sub = all_results[all_results["dataset"] == ds_name]
        weighted_r2 = np.average(sub["r2"], weights=sub["n_samples"])
        print(
            f"  {ds_name}: 加权 R2={weighted_r2:.6f}, "
            f"平均 R2={sub['r2'].mean():.6f}, "
            f"平均 RMSE={sub['rmse'].mean():.6f}, "
            f"平均 MAE={sub['mae'].mean():.6f}"
        )

    if global_y is not None and global_p is not None:
        pooled_r2 = r2_score(global_y, global_p)
        pooled_rmse = np.sqrt(mean_squared_error(global_y, global_p))
        pooled_mae = mean_absolute_error(global_y, global_p)
        print(f"\n{'='*70}")
        print(f" 全局 Pooled 指标 (所有样本合并, n={len(global_y)})")
        print(f"{'='*70}")
        print(f"  Pooled R2:   {pooled_r2:.6f}")
        print(f"  Pooled RMSE: {pooled_rmse:.6f}")
        print(f"  Pooled MAE:  {pooled_mae:.6f}")

        # 追加全局 pooled 行到 all_results
        global_row = pd.DataFrame([{
            "dataset": "global",
            "rxntype": "pooled",
            "n_samples": len(global_y),
            "r2": pooled_r2,
            "rmse": pooled_rmse,
            "mae": pooled_mae,
        }])
        all_results = pd.concat([all_results, global_row], ignore_index=True)

    return all_results


# ──────────────────────────────────────
# 主流程
# ──────────────────────────────────────
def main():
    round1_path = SCRIPT_DIR.parent.parent / "data" / "round1_test_data_with_ans.csv"
    round2_path = SCRIPT_DIR.parent.parent / "data" / "round2_test_data_with_ans.csv"

    all_results = []
    global_y_list = []
    global_p_list = []

    if round1_path.exists():
        df1 = pd.read_csv(round1_path)
        df1["rxntype"] = 1
        res1, y1, p1 = evaluate_dataset(df1, "round1")
        all_results.append(res1)
        global_y_list.append(y1)
        global_p_list.append(p1)
    else:
        print(f"[跳过] 找不到文件: {round1_path}")

    if round2_path.exists():
        df2 = pd.read_csv(round2_path)
        df2["rxntype"] = pd.to_numeric(df2["rxntype"], errors="coerce").astype(int)
        res2, y2, p2 = evaluate_dataset(df2, "round2")
        all_results.append(res2)
        global_y_list.append(y2)
        global_p_list.append(p2)
    else:
        print(f"[跳过] 找不到文件: {round2_path}")

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        global_y = np.concatenate(global_y_list)
        global_p = np.concatenate(global_p_list)
        all_results_df = print_summary(all_results_df, global_y, global_p)

        out_path = SCRIPT_DIR.parent / "results" / "avalon_rdkit_optuna_test_results.csv"
        all_results_df.to_csv(out_path, index=False)
        print(f"\n结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
