# generate_test_rdkit_gz.py
# 生成测试集的 RDKit 特征 gz 文件，格式与 train-rdkitfeature-rxn{N}.gz 完全一致
#
# 查找策略（与训练集 gz 生成逻辑对齐）：
#   Reactant1, Reactant2 → train-rdkitfeature-{col}.json
#   Product              → train-rdkitfeature-Product.json（混合 SMILES 拆分取均值）
#   Additive             → train-rdkitfeature-Additive-nosplit.json
#   Solvent              → train-rdkitfeature-Solvent-nosplit.json
# 查不到的 SMILES → CalcMolDescriptors 实时计算
#
# 用法: python generate_test_rdkit_gz.py
# 输出: extra-rdkit/test-rdkitfeature-rxn{N}.gz

import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RDKIT_DIR = DATASET_DIR / "extra-rdkit"

# 描述符名称列表（与训练 gz 一致，210 个）
FEATURE_NAMES_CSV = RDKIT_DIR / "train-rdkitfeature-Reactant1_feature_names.csv"
names_df = pd.read_csv(FEATURE_NAMES_CSV)
DESC_NAMES = names_df["FeatureName"].tolist()
N_DESC = len(DESC_NAMES)  # 210


# ──────────────────────────────────────
# 描述符计算（按 DESC_NAMES 顺序）
# ──────────────────────────────────────
def compute_descriptor_vec(mol: Chem.Mol) -> list[float]:
    vals = Descriptors.CalcMolDescriptors(mol)
    vec = []
    for name in DESC_NAMES:
        v = vals.get(name, 0.0)
        if v is None or not math.isfinite(v):
            v = 0.0
        vec.append(float(v))
    return vec


# ──────────────────────────────────────
# 加载 JSON 查找表
# ──────────────────────────────────────
def load_lookups() -> dict[str, dict]:
    """按列加载 JSON 查找表。
    Additive/Solvent 用 nosplit 版本（与训练 gz 一致）。"""
    lookups = {}
    for col in ["Reactant1", "Reactant2", "Product"]:
        path = RDKIT_DIR / f"train-rdkitfeature-{col}.json"
        with open(path) as f:
            lookups[col] = json.load(f)
        print(f"  {col}: {len(lookups[col])} entries (split JSON)")
    for col in ["Additive", "Solvent"]:
        path = RDKIT_DIR / f"train-rdkitfeature-{col}-nosplit.json"
        with open(path) as f:
            lookups[col] = json.load(f)
        print(f"  {col}: {len(lookups[col])} entries (nosplit JSON)")
    return lookups


# ──────────────────────────────────────
# 获取单个 SMILES 的特征向量
# ──────────────────────────────────────
def get_feature_vec(smi: str, col: str, lookup: dict) -> list[float]:
    """查表 → 计算，返回 N_DESC 维向量。"""
    if smi in lookup:
        return lookup[smi][:N_DESC]

    # Additive/Solvent 用 nosplit JSON，不需要拆分
    if col in ("Additive", "Solvent"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return compute_descriptor_vec(mol)
        return [0.0] * N_DESC

    # Reactant1/Reactant2/Product：混合 SMILES 拆分取均值
    parts = [p.strip() for p in smi.split(".") if p.strip()]
    if not parts:
        return [0.0] * N_DESC

    part_vecs = []
    for part in parts:
        if part in lookup:
            part_vecs.append(lookup[part][:N_DESC])
        else:
            mol = Chem.MolFromSmiles(part)
            if mol is not None:
                part_vecs.append(compute_descriptor_vec(mol))
            else:
                part_vecs.append([0.0] * N_DESC)

    if len(part_vecs) == 1:
        return part_vecs[0]
    return np.nanmean(part_vecs, axis=0).tolist()


# ──────────────────────────────────────
# 为一组样本生成完整特征矩阵
# ──────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame, lookups: dict) -> np.ndarray:
    """5 列拼接 → (n_samples, 1050)"""
    all_cols = []
    for col in MOLECULE_COLUMNS:
        vecs = []
        for smi in df[col].fillna("").astype(str):
            vecs.append(get_feature_vec(smi, col, lookups[col]))
        arr = np.asarray(vecs, dtype=np.float64)
        # 钳制到 float32 安全范围，避免溢出
        arr = np.clip(arr, -3.4e38, 3.4e38)
        all_cols.append(arr.astype(np.float32))
    return np.concatenate(all_cols, axis=1)


# ──────────────────────────────────────
# 写入 gz 文件（与训练 gz 格式一致）
# ──────────────────────────────────────
def save_gz(arr: np.ndarray, path: Path) -> None:
    lines = []
    for row in arr:
        lines.append(",".join(f"{v}" for v in row))
    content = "\n".join(lines) + "\n"
    with gzip.open(path, "wb") as f:
        f.write(content.encode())
    print(f"  保存: {path} ({arr.shape[0]} rows, {arr.shape[1]} features)")


# ──────────────────────────────────────
# 主流程
# ──────────────────────────────────────
def main():
    print("=" * 60)
    print("生成测试集 RDKit 特征 gz 文件")
    print("=" * 60)

    print(f"描述符数: {N_DESC}")
    print(f"输出目录: {RDKIT_DIR}")

    # 加载 JSON 查找表
    print("\n[1/3] 加载训练集 JSON 查找表...")
    lookups = load_lookups()

    # 加载测试数据
    print("\n[2/3] 加载测试数据...")
    test_dfs = {}
    r1_path = DATASET_DIR / "round1_test_data_with_ans.csv"
    r2_path = DATASET_DIR / "round2_test_data_with_ans.csv"
    if r1_path.exists():
        df1 = pd.read_csv(r1_path).copy()
        df1["rxntype"] = 1
        test_dfs[1] = df1
        print(f"  Round1: {len(df1)} 样本 (rxntype=1)")
    if r2_path.exists():
        df2 = pd.read_csv(r2_path).copy()
        if "rxntype" not in df2.columns:
            df2["rxntype"] = 2
        for rt in sorted(df2["rxntype"].unique()):
            sub = df2[df2["rxntype"] == rt].reset_index(drop=True)
            test_dfs[int(rt)] = sub
            print(f"  Round2 rxntype={int(rt)}: {len(sub)} 样本")

    # 按 rxntype 生成 gz
    print("\n[3/3] 生成 gz 文件...")

    # 统计查表覆盖率
    hit_total = 0
    miss_total = 0

    for rxntype in sorted(test_dfs.keys()):
        df = test_dfs[rxntype]
        print(f"\n  rxntype={rxntype}, n={len(df)}")

        # 统计查表命中率
        for col in MOLECULE_COLUMNS:
            smis = df[col].fillna("").astype(str).unique()
            col_lookup = lookups[col]
            hit = sum(1 for s in smis if s in col_lookup)
            miss = len(smis) - hit
            hit_total += hit
            miss_total += miss
            if miss > 0:
                print(f"    {col}: {hit}/{len(smis)} 查表命中, {miss} 需实时计算")

        # 生成特征矩阵
        feat_matrix = build_feature_matrix(df, lookups)

        # NaN/Inf 处理
        nan_mask = np.isnan(feat_matrix)
        inf_mask = np.isinf(feat_matrix)
        problem_mask = nan_mask | inf_mask
        if problem_mask.any():
            col_means = np.nanmean(feat_matrix, axis=0)
            col_means[np.isnan(col_means)] = 0.0
            for j in range(feat_matrix.shape[1]):
                mask = problem_mask[:, j]
                if mask.any():
                    feat_matrix[mask, j] = col_means[j]
            print(f"    修复了 {problem_mask.sum()} 个 NaN/Inf 值")

        # 保存 gz
        out_path = RDKIT_DIR / f"test-rdkitfeature-rxn{rxntype}.gz"
        save_gz(feat_matrix, out_path)

    print(f"\n查表统计: {hit_total} 命中, {miss_total} 实时计算")
    print(f"覆盖率: {hit_total / (hit_total + miss_total) * 100:.1f}%")
    print("\n完成！")


if __name__ == "__main__":
    main()
