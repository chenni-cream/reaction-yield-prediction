"""
testavalon.py
使用 Avalon 指纹 + LightGBM 模型对 round1 / round2 测试集进行泛化性验证。
模型来源: model_selection/ckpt-avalon-fp-comparison/LightGBM/rxn_{type}/LightGBM_fold{i}.txt
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Avalon import pyAvalonTools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048
N_FOLDS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent / "ckpt-avalon-fp-comparison" / "LightGBM"


# ──────────────────────────────────────
# 特征工程（与训练时完全一致）
# ──────────────────────────────────────
def bitvect_to_numpy(bitvect: DataStructs.ExplicitBitVect, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def avalon_fp(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
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
# 模型加载与预测
# ──────────────────────────────────────
def predict_with_ensemble(X: np.ndarray, rxntype: int) -> np.ndarray:
    """加载 rxntype 对应的 5 折模型，取预测均值作为集成预测。"""
    rxn_dir = MODEL_ROOT / f"rxn_{rxntype}"
    if not rxn_dir.exists():
        raise FileNotFoundError(f"找不到模型目录: {rxn_dir}")

    preds_list = []
    for fold in range(1, N_FOLDS + 1):
        model_path = rxn_dir / f"LightGBM_fold{fold}.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        booster = lgb.Booster(model_file=str(model_path))
        preds_list.append(booster.predict(X))

    return np.mean(preds_list, axis=0)


# ──────────────────────────────────────
# 评估逻辑
# ──────────────────────────────────────
def evaluate_dataset(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """对单个数据集按 rxntype 分别预测并评估。
    返回 (分rxntype结果DataFrame, 全部真实值, 全部预测值)。"""
    print(f"\n{'='*70}")
    print(f" 数据集: {dataset_name}  |  样本数: {len(df)}")
    print(f"{'='*70}")

    results = []
    all_trues = []
    all_preds = []

    for rxntype in sorted(df["rxntype"].unique()):
        sub = df[df["rxntype"] == rxntype].reset_index(drop=True)
        print(f"\n  rxntype={rxntype}, n={len(sub)}")

        X = build_features(sub)
        y_true = sub[TARGET_COLUMN].to_numpy(dtype=np.float32)

        preds = predict_with_ensemble(X, rxntype)

        all_trues.append(y_true)
        all_preds.append(preds)

        r2 = r2_score(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)

        results.append(
            {
                "dataset": dataset_name,
                "rxntype": rxntype,
                "n_samples": len(sub),
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
            }
        )
        print(f"    R2: {r2:.6f}  |  RMSE: {rmse:.6f}  |  MAE: {mae:.6f}")

    return pd.DataFrame(results), np.concatenate(all_trues), np.concatenate(all_preds)


def print_summary(all_results: pd.DataFrame, global_y=None, global_p=None) -> pd.DataFrame:
    """打印汇总表格和整体指标，返回含 global pooled 行的 DataFrame。"""
    print(f"\n{'='*70}")
    print(" 汇总结果")
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

    # 按 dataset 汇总整体指标
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
def main() -> None:
    round1_path = SCRIPT_DIR.parent.parent / "data" / "round1_test_data_with_ans.csv"
    round2_path = SCRIPT_DIR.parent.parent / "data" / "round2_test_data_with_ans.csv"

    all_results = []
    all_global_trues = []
    all_global_preds = []

    # ── Round 1 ──
    if round1_path.exists():
        df1 = pd.read_csv(round1_path)
        df1["rxntype"] = 1  # round1 全部为 rxntype=1
        res1, t1, p1 = evaluate_dataset(df1, "round1")
        all_results.append(res1)
        all_global_trues.append(t1)
        all_global_preds.append(p1)
    else:
        print(f"[跳过] 找不到文件: {round1_path}")

    # ── Round 2 ──
    if round2_path.exists():
        df2 = pd.read_csv(round2_path)
        df2["rxntype"] = pd.to_numeric(df2["rxntype"], errors="coerce").astype(int)
        res2, t2, p2 = evaluate_dataset(df2, "round2")
        all_results.append(res2)
        all_global_trues.append(t2)
        all_global_preds.append(p2)
    else:
        print(f"[跳过] 找不到文件: {round2_path}")

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)

        global_y = np.concatenate(all_global_trues) if all_global_trues else None
        global_p = np.concatenate(all_global_preds) if all_global_preds else None
        all_results_df = print_summary(all_results_df, global_y, global_p)

        # 保存结果
        out_path = SCRIPT_DIR.parent / "results" / "test_avalon_results.csv"
        all_results_df.to_csv(out_path, index=False)
        print(f"\n结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
