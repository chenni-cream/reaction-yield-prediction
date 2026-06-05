"""
layer-RDKit-test.py
使用按 rxntype 分类的 LightGBM 融合模型 (LayeredFingerprint + RDKit Descriptors) 对 round1 / round2 测试集进行泛化性验证。
模型来源: All-reactions/ckpt-layered-rdkit-fusion/model_selection/fusion/rxn_{type}/lgbm_fusion_fold{i}.txt
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048
N_FOLDS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = (
    SCRIPT_DIR
    / "ckpt-layered-rdkit-fusion"
    / "fusion"
)


# ──────────────────────────────────────
# Layered 指纹特征（与训练时一致）
# ──────────────────────────────────────
def bitvect_to_numpy(bitvect: DataStructs.ExplicitBitVect, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def layered_fp(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    fp = Chem.LayeredFingerprint(mol, fpSize=n_bits)
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
            rows.append(layered_fp(mol, FP_SIZE))
    return np.asarray(rows, dtype=np.float32)


def build_layered_features(df: pd.DataFrame) -> np.ndarray:
    feats = [encode_smiles_column(df[col]) for col in MOLECULE_COLUMNS]
    return np.concatenate(feats, axis=1)


# ──────────────────────────────────────
# RDKit 描述符特征：查表 + 实时计算（均按训练集 gz 文件的实际描述符顺序）
# ──────────────────────────────────────
RDKIT_DIR = SCRIPT_DIR.parent.parent / "data" / "extra-rdkit"

# gz 文件的实际描述符顺序（已通过向量相关性验证 210/210 匹配）
_GZ_ORDER_DF = pd.read_csv(RDKIT_DIR / "gz_descriptor_order.csv")
_GZ_DESC_NAMES = _GZ_ORDER_DF["name"].tolist()
N_RDKIT_DESC = len(_GZ_DESC_NAMES)  # 210

# JSON 查找优先级：测试集 JSON > 训练集 JSON（nosplit > split）
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


def _load_json_lookup(filenames: list[str]) -> dict[str, list[float]] | None:
    """按优先级依次尝试加载 JSON，返回第一个成功加载的字典，全部失败则返回 None。"""
    for fname in filenames:
        path = RDKIT_DIR / fname
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    return None


def _compute_rdkit_vec(mol: Chem.Mol) -> np.ndarray:
    """用 CalcMolDescriptors 按 gz 实际顺序计算描述符向量。"""
    from rdkit.Chem import Descriptors as Desc
    vals = Desc.CalcMolDescriptors(mol)
    vec = []
    for name in _GZ_DESC_NAMES:
        v = vals.get(name, 0.0)
        if v is None or not np.isfinite(v):
            v = 0.0
        vec.append(float(v))
    return np.array(vec, dtype=np.float32)


def compute_rdkit_descriptors_for_column(
    smiles_series: pd.Series,
    col_name: str,
) -> np.ndarray:
    """对一列 SMILES 获取 RDKit 描述符。

    策略：优先从 JSON 查找表获取，未命中时用 CalcMolDescriptors
    按 gz 实际顺序实时计算，保证与训练时特征完全一致。
    """
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


def build_rdkit_features(df: pd.DataFrame) -> np.ndarray:
    """为 5 列分子分别获取 RDKit 描述符并拼接。"""
    feats = [
        compute_rdkit_descriptors_for_column(df[col], col)
        for col in MOLECULE_COLUMNS
    ]
    return np.concatenate(feats, axis=1)


# ──────────────────────────────────────
# 融合特征构建
# ──────────────────────────────────────
def build_features(df: pd.DataFrame) -> np.ndarray:
    """拼接 Layered FP + RDKit 描述符，与训练时特征维度一致。"""
    layered = build_layered_features(df)
    rdkit = build_rdkit_features(df)
    return np.concatenate([layered, rdkit], axis=1)


# ──────────────────────────────────────
# 模型加载与预测
# ──────────────────────────────────────
def predict_with_ensemble(X: np.ndarray, rxntype: int) -> np.ndarray:
    """加载 rxntype 对应的 5 折融合模型，取预测均值作为集成预测。"""
    rxn_dir = MODEL_ROOT / f"rxn_{rxntype}"
    if not rxn_dir.exists():
        raise FileNotFoundError(f"找不到模型目录: {rxn_dir}")

    preds_list = []
    for fold in range(1, N_FOLDS + 1):
        model_path = rxn_dir / f"lgbm_fusion_fold{fold}.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        booster = lgb.Booster(model_file=str(model_path))
        preds_list.append(booster.predict(X))

    return np.mean(preds_list, axis=0)


# ──────────────────────────────────────
# 评估逻辑
# ──────────────────────────────────────
def evaluate_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """对单个数据集按 rxntype 分别预测并评估。"""
    print(f"\n{'='*70}")
    print(f" 数据集: {dataset_name}  |  样本数: {len(df)}")
    print(f"{'='*70}")

    results = []

    for rxntype in sorted(df["rxntype"].unique()):
        sub = df[df["rxntype"] == rxntype].reset_index(drop=True)
        print(f"\n  rxntype={rxntype}, n={len(sub)}")

        X = build_features(sub)
        y_true = sub[TARGET_COLUMN].to_numpy(dtype=np.float32)

        print(f"    特征维度: {X.shape[1]}")

        preds = predict_with_ensemble(X, rxntype)

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

    return pd.DataFrame(results)


def print_summary(all_results: pd.DataFrame) -> None:
    """打印汇总表格和整体指标。"""
    print(f"\n{'='*70}")
    print(" 汇总结果 (Layered FP + RDKit Descriptors 融合模型)")
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


# ──────────────────────────────────────
# 主流程
# ──────────────────────────────────────
def main() -> None:
    round1_path = SCRIPT_DIR.parent.parent / "data" / "round1_test_data_with_ans.csv"
    round2_path = SCRIPT_DIR.parent.parent / "data" / "round2_test_data_with_ans.csv"

    all_results = []

    # ── Round 1 ──
    if round1_path.exists():
        df1 = pd.read_csv(round1_path)
        df1["rxntype"] = 1  # round1 全部为 rxntype=1
        res1 = evaluate_dataset(df1, "round1")
        all_results.append(res1)
    else:
        print(f"[跳过] 找不到文件: {round1_path}")

    # ── Round 2 ──
    if round2_path.exists():
        df2 = pd.read_csv(round2_path)
        df2["rxntype"] = pd.to_numeric(df2["rxntype"], errors="coerce").astype(int)
        res2 = evaluate_dataset(df2, "round2")
        all_results.append(res2)
    else:
        print(f"[跳过] 找不到文件: {round2_path}")

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        print_summary(all_results_df)

        # 保存结果
        out_path = SCRIPT_DIR.parent / "results" / "layer_rdkit_fusion_test_results.csv"
        all_results_df.to_csv(out_path, index=False)
        print(f"\n结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
