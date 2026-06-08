"""
layer-RDKit-qc-solvent-test.py
The LightGBM ensemble model classified by rxntype (LayeredFingerprint + RDKit + QC + Solvent) was used to conduct generalization verification on the round1 / round2 test sets.
The top-500 non-fingerprint features of each rxntype were read from the corresponding feature_importance_full.csv file.
"""

import gzip
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048
N_RDKIT = 210
N_FOLDS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent.parent / "data"
MODEL_ROOT = SCRIPT_DIR.parent / "ckpt-ablation-solvent-qc"

# ──────────────────────────────────────
# 溶剂特征列定义（与训练一致，3 源去重后 31 维）
# ──────────────────────────────────────
SOLV_MAIN_COLS = [
    "MW (g/mol)", "Density (g/mL)", "Molar volume (mL/mol)", "Refractive index",
    "Mol. refr. pow. (mL/mol)", "Dipole moment (D)", "Melting point (°C)",
    "Boiling point (°C)", "Viscosity (cP)", "lnP (partition coeff.)",
    "Vapour pressure (mbar)", "Henry's constant", "lngamma", "neutral",
]
MNSOL_COLS = ["alpha", "beta", "beta**2", "eps", "gamma", "n", "phi**2", "psi**2"]
DRUGBANK_COLS = [
    "logP_ALOGPS", "logS", "logP_ChemAxon", "pKa_acid", "pKa_base",
    "PSA", "Polarizability", "H_Acceptor_Count", "H_Donor_Count",
]
ALL_SOLV_COLS = SOLV_MAIN_COLS + MNSOL_COLS + DRUGBANK_COLS
N_SOLV_FEATURES = len(ALL_SOLV_COLS)  # 31

# ──────────────────────────────────────
# QC 分子描述符列
# ──────────────────────────────────────
QC_FEATURE_COLS = [
    "dipole", "electron_affinity", "electrophilicity", "nucleophilicity",
    "electrofugality", "nucleofugality", "homo", "lumo", "homo-lumo",
    "ionization_potential",
]
N_QC_FEATURES = len(QC_FEATURE_COLS)  # 10

# 全量特征维度
N_LAYERED = len(MOLECULE_COLUMNS) * FP_SIZE      # 10240
N_RDKIT_TOTAL = len(MOLECULE_COLUMNS) * N_RDKIT   # 1050


# ═══════════════════════════════════════
# 1. Layered 指纹特征
# ═══════════════════════════════════════
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
        rows.append(zero if mol is None else layered_fp(mol, FP_SIZE))
    return np.asarray(rows, dtype=np.float32)


def build_layered_features(df: pd.DataFrame) -> np.ndarray:
    return np.concatenate([encode_smiles_column(df[col]) for col in MOLECULE_COLUMNS], axis=1)


# ═══════════════════════════════════════
# 2. RDKit 描述符特征（JSON 查找 + 实时计算）
# ═══════════════════════════════════════
RDKIT_DIR = DATASET_DIR / "extra-rdkit"

_JSON_PRIORITY = {
    "Reactant1": ["train-rdkitfeature-Reactant1.json"],
    "Reactant2": ["train-rdkitfeature-Reactant2.json"],
    "Product":   ["train-rdkitfeature-Product.json"],
    "Additive":  ["train-rdkitfeature-Additive-nosplit.json",
                   "train-rdkitfeature-Additive.json"],
    "Solvent":   ["train-rdkitfeature-Solvent-nosplit.json",
                   "train-rdkitfeature-Solvent.json"],
}


def _load_json_lookup(filenames: list[str]) -> dict[str, list[float]] | None:
    for fname in filenames:
        path = RDKIT_DIR / fname
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    return None


def _get_descriptor_calculator() -> MoleculeDescriptors.MolecularDescriptorCalculator:
    feature_names_path = RDKIT_DIR / "train-rdkitfeature-Reactant1_feature_names.csv"
    names_df = pd.read_csv(feature_names_path)
    desc_names = names_df["FeatureName"].tolist()
    return MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)


def compute_rdkit_descriptors_for_column(
    smiles_series: pd.Series,
    col_name: str,
    calc: MoleculeDescriptors.MolecularDescriptorCalculator,
    n_desc: int,
) -> np.ndarray:
    smiles = smiles_series.fillna("").astype(str).tolist()
    zero = np.zeros((n_desc,), dtype=np.float32)
    lookup = _load_json_lookup(_JSON_PRIORITY.get(col_name, []))

    rows = []
    hit_count = 0
    miss_count = 0

    for smi in smiles:
        if lookup is not None and smi in lookup:
            arr = np.array(lookup[smi], dtype=np.float32)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            rows.append(arr)
            hit_count += 1
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append(zero)
            else:
                vals = calc.CalcDescriptors(mol)
                arr = np.array(vals, dtype=np.float32)
                arr = np.where(np.isfinite(arr), arr, 0.0)
                rows.append(arr)
            miss_count += 1

    source = "JSON" if lookup is not None else "计算"
    print(f"    {col_name}: 查表命中 {hit_count}, 实时计算 {miss_count} (来源: {source})")
    return np.asarray(rows, dtype=np.float32)


def build_rdkit_features(df: pd.DataFrame) -> np.ndarray:
    calc = _get_descriptor_calculator()
    n_desc = len(calc.GetDescriptorNames())
    feats = [
        compute_rdkit_descriptors_for_column(df[col], col, calc, n_desc)
        for col in MOLECULE_COLUMNS
    ]
    return np.concatenate(feats, axis=1)


# ═══════════════════════════════════════
# 3. 溶剂描述符特征
# ═══════════════════════════════════════
def build_solvent_lookup(solvents_dir: Path, drugbank_dir: Path) -> dict[str, np.ndarray]:
    main_df = pd.read_csv(solvents_dir / "solvent_withsmiles.csv")
    main_df = main_df.dropna(subset=["smiles"])

    mnsol_df = pd.read_csv(solvents_dir / "MNSol_alldata_withsmiles.csv")
    mnsol_df = mnsol_df.dropna(subset=["smiles"])

    drug_df = pd.read_csv(drugbank_dir / "solvent.csv")
    drug_df = drug_df.dropna(subset=["smiles"])

    merged = main_df[["smiles"] + SOLV_MAIN_COLS].copy()
    merged = merged.merge(mnsol_df[["smiles"] + MNSOL_COLS], on="smiles", how="outer")
    merged = merged.merge(drug_df[["smiles"] + DRUGBANK_COLS], on="smiles", how="outer")

    lookup = {}
    for _, row in merged.iterrows():
        smi = row["smiles"]
        if pd.isna(smi):
            continue
        lookup[str(smi)] = row[ALL_SOLV_COLS].to_numpy(dtype=np.float64)

    # 用列均值填充 NaN
    arr_all = np.array(list(lookup.values()))
    col_means = np.nanmean(arr_all, axis=0)
    col_means[np.isnan(col_means)] = 0.0
    for smi in lookup:
        vec = lookup[smi]
        nan_mask = np.isnan(vec)
        if nan_mask.any():
            vec[nan_mask] = col_means[nan_mask]
            lookup[smi] = vec

    print(f"  溶剂查找表: {len(lookup)} 条 SMILES, {N_SOLV_FEATURES} 维特征")
    return lookup


def build_solvent_features(df: pd.DataFrame, solvent_lookup: dict) -> np.ndarray:
    zero_vec = np.zeros(N_SOLV_FEATURES, dtype=np.float32)
    col_feats = []

    for smi in df["Solvent"].fillna("").astype(str):
        parts = smi.split(".")
        vecs = []
        for part in parts:
            part = part.strip()
            if part in solvent_lookup:
                vecs.append(solvent_lookup[part].astype(np.float32))
        col_feats.append(np.mean(vecs, axis=0) if vecs else zero_vec)

    arr = np.asarray(col_feats, dtype=np.float32)

    problem_mask = np.isnan(arr) | np.isinf(arr)
    if problem_mask.any():
        col_means = np.nanmean(arr, axis=0)
        col_means[np.isnan(col_means)] = 0.0
        for j in range(arr.shape[1]):
            mask = problem_mask[:, j]
            if mask.any():
                arr[mask, j] = col_means[j]

    return arr


# ═══════════════════════════════════════
# 4. QC 分子描述符特征
# ═══════════════════════════════════════
def _load_qc_gz(gz_path: Path) -> pd.DataFrame:
    with gzip.open(gz_path, "rt") as f:
        return pd.read_csv(f)


def build_qc_lookup() -> dict[str, dict[str, np.ndarray]]:
    """从训练/测试 QC 数据构建查找表，按分子列分组。"""
    file_map = {
        "Reactant1": "psikit_Reactant1.csv.gz",
        "Reactant2": "psikit_Reactant2.csv.gz",
        "Product":   "psikit_Product.csv.gz",
        "Additive":  "psikit_Additive.csv.gz",
        "Solvent":   "psikit_Solvent.csv.gz",
    }

    # 搜索所有 QC 数据目录
    qc_dirs = [
        DATASET_DIR / "qm_desc-morfeus-round1",        # 训练 round1（无 Product）
        DATASET_DIR / "qm_desc-morfeus",                # 训练 round2
        DATASET_DIR / "qm_desc-morfeus-round1-test",    # 测试 round1
        DATASET_DIR / "qm_desc-morfeus-round2-test",    # 测试 round2
    ]

    lookups = {}
    for col, filename in file_map.items():
        lookup = {}
        for qc_dir in qc_dirs:
            if not qc_dir.exists():
                continue
            # Round1 training data does not contain Product QC descriptors
            if col == "Product" and qc_dir == DATASET_DIR / "qm_desc-morfeus-round1":
                continue
            gz_path = qc_dir / filename
            if not gz_path.exists():
                continue
            df_qc = _load_qc_gz(gz_path)
            for _, row in df_qc.iterrows():
                smi = row["smile"]
                if smi not in lookup:  # 不覆盖已有值
                    lookup[smi] = row[QC_FEATURE_COLS].to_numpy(dtype=np.float32)

        lookups[col] = lookup
        print(f"  QC lookup '{col}': {len(lookup)} unique SMILES")

    return lookups


def build_qc_features(df: pd.DataFrame, qc_lookups: dict) -> np.ndarray:
    parts = []
    zero_vec = np.zeros(N_QC_FEATURES, dtype=np.float32)

    for col in MOLECULE_COLUMNS:
        lookup = qc_lookups[col]
        col_feats = []
        for smi in df[col].fillna("").astype(str):
            col_feats.append(lookup.get(smi, zero_vec))
        parts.append(np.asarray(col_feats, dtype=np.float32))

    arr = np.concatenate(parts, axis=1)

    problem_mask = np.isnan(arr) | np.isinf(arr)
    if problem_mask.any():
        col_means = np.nanmean(arr, axis=0)
        col_means[np.isnan(col_means)] = 0.0
        for j in range(arr.shape[1]):
            mask = problem_mask[:, j]
            if mask.any():
                arr[mask, j] = col_means[j]

    return arr


# ═══════════════════════════════════════
# 5. 特征选择：根据 importance CSV 选取 top-500 非指纹特征
# ═══════════════════════════════════════
def _parse_feature_to_global_idx(feat_name: str) -> int:
    """将特征名映射为全量特征矩阵中的列索引。

    全量顺序: [Layered(10240)] + [RDKit(1050)] + [Solvent(31)] + [QC(50)]
    """
    if feat_name.startswith("Layered_"):
        # Layered_{col}_{i} — 不应出现在非指纹特征中
        parts = feat_name.split("_", 2)
        col = parts[1]
        idx = int(parts[2])
        return MOLECULE_COLUMNS.index(col) * FP_SIZE + idx

    if feat_name.startswith("RDKit_"):
        # RDKit_{col}_{i}
        parts = feat_name.split("_", 2)
        col = parts[1]
        idx = int(parts[2])
        return N_LAYERED + MOLECULE_COLUMNS.index(col) * N_RDKIT + idx

    if feat_name.startswith("Solvent_"):
        # Solvent_{prop_name}
        prop = feat_name[len("Solvent_"):]
        return N_LAYERED + N_RDKIT_TOTAL + ALL_SOLV_COLS.index(prop)

    if feat_name.startswith("QC_"):
        # QC_{col}_{desc}
        parts = feat_name.split("_", 2)
        col = parts[1]
        desc = parts[2]
        return N_LAYERED + N_RDKIT_TOTAL + N_SOLV_FEATURES + \
               MOLECULE_COLUMNS.index(col) * N_QC_FEATURES + QC_FEATURE_COLS.index(desc)

    raise ValueError(f"未知特征名格式: {feat_name}")


def get_selected_indices(rxntype: int, top_k: int = 500) -> np.ndarray:
    """读取 rxntype 的 feature_importance_full.csv，返回筛选后的列索引（已排序）。"""
    csv_path = MODEL_ROOT / f"rxn_{rxntype}" / "feature_importance_full.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到特征重要性文件: {csv_path}")

    imp_df = pd.read_csv(csv_path)

    # 提取非 Layered 特征（RDKit + Solvent + QC）
    nonfp = imp_df[~imp_df["feature"].str.startswith("Layered_")].copy()
    nonfp = nonfp[nonfp["importance"] > 0].reset_index(drop=True)

    # 取 top_k 个非指纹特征的全局列索引
    k_use = min(top_k, len(nonfp))
    selected = []
    for feat_name in nonfp["feature"].iloc[:k_use]:
        selected.append(_parse_feature_to_global_idx(feat_name))

    # 合并 Layered 列 + 选中的非指纹列，排序
    fixed = np.arange(N_LAYERED)
    all_indices = np.concatenate([fixed, np.array(selected)])
    all_indices.sort()
    return all_indices


# ═══════════════════════════════════════
# 6. 模型加载与预测
# ═══════════════════════════════════════
def predict_with_ensemble(X: np.ndarray, rxntype: int) -> np.ndarray:
    """加载 rxntype 对应的 5 折模型，取预测均值作为集成预测。"""
    model_dir = MODEL_ROOT / f"rxn_{rxntype}" / "best_k_500"
    if not model_dir.exists():
        raise FileNotFoundError(f"找不到模型目录: {model_dir}")

    preds_list = []
    for fold in range(1, N_FOLDS + 1):
        model_path = model_dir / f"fold{fold}.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        booster = lgb.Booster(model_file=str(model_path))
        preds_list.append(booster.predict(X))

    return np.mean(preds_list, axis=0)


# ═══════════════════════════════════════
# 7. 评估逻辑
# ═══════════════════════════════════════
def evaluate_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    solvent_lookup: dict,
    qc_lookups: dict,
) -> pd.DataFrame:
    """对单个数据集按 rxntype 分别预测并评估。"""
    print(f"\n{'='*70}")
    print(f" 数据集: {dataset_name}  |  样本数: {len(df)}")
    print(f"{'='*70}")

    results = []
    all_trues = []
    all_preds = []

    # 构建全量特征（所有 rxntype 共享底层特征）
    print("  构建全量特征矩阵...")
    layered = build_layered_features(df)
    rdkit = build_rdkit_features(df)
    solv = build_solvent_features(df, solvent_lookup)
    qc = build_qc_features(df, qc_lookups)
    X_full = np.concatenate([layered, rdkit, solv, qc], axis=1)
    print(f"  全量特征维度: {X_full.shape}")

    for rxntype in sorted(df["rxntype"].unique()):
        sub_mask = df["rxntype"].values == rxntype
        n_samples = sub_mask.sum()
        print(f"\n  rxntype={rxntype}, n={n_samples}")

        # 特征选择
        selected_indices = get_selected_indices(rxntype, top_k=500)
        X_selected = X_full[sub_mask][:, selected_indices]
        print(f"    选中特征维度: {X_selected.shape[1]}")

        y_true = df.loc[sub_mask, TARGET_COLUMN].to_numpy(dtype=np.float32)

        preds = predict_with_ensemble(X_selected, rxntype)

        r2 = r2_score(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)

        results.append({
            "dataset": dataset_name,
            "rxntype": rxntype,
            "n_samples": n_samples,
            "n_features": X_selected.shape[1],
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
        })
        print(f"    R2: {r2:.6f}  |  RMSE: {rmse:.6f}  |  MAE: {mae:.6f}")

        all_trues.append(y_true)
        all_preds.append(preds)

    # pooled R²
    all_trues = np.concatenate(all_trues)
    all_preds = np.concatenate(all_preds)
    pooled_r2 = r2_score(all_trues, all_preds)
    print(f"\n  >> {dataset_name} pooled R² = {pooled_r2:.6f}")

    return pd.DataFrame(results), all_trues, all_preds


def print_summary(all_results: pd.DataFrame) -> None:
    """打印汇总表格和整体指标。"""
    print(f"\n{'='*70}")
    print(" 汇总结果 (Layered FP + RDKit + QC + Solvent 融合模型, Top-500)")
    print(f"{'='*70}")
    print(
        f"{'数据集':<12} {'rxntype':<10} {'样本数':<10} "
        f"{'特征数':<10} {'R2':<14} {'RMSE':<14} {'MAE':<14}"
    )
    print("-" * 80)
    for _, row in all_results.iterrows():
        print(
            f"{row['dataset']:<12} {int(row['rxntype']):<10} {int(row['n_samples']):<10} "
            f"{int(row['n_features']):<10} "
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


# ═══════════════════════════════════════
# 主流程
# ═══════════════════════════════════════
def main() -> None:
    round1_path = SCRIPT_DIR.parent.parent / "data" / "round1_test_data_with_ans.csv"
    round2_path = SCRIPT_DIR.parent.parent / "data" / "round2_test_data_with_ans.csv"

    # ── 加载溶剂描述符查找表 ──
    print("加载溶剂描述符...")
    solvent_lookup = build_solvent_lookup(
        DATASET_DIR / "solvents",
        DATASET_DIR / "drugbank",
    )

    # ── 加载 QC 描述符查找表 ──
    print("\n加载 QC 分子描述符...")
    qc_lookups = build_qc_lookup()

    all_results = []
    all_global_trues = []
    all_global_preds = []

    # ── Round 1 ──
    if round1_path.exists():
        df1 = pd.read_csv(round1_path)
        df1["rxntype"] = 1  # round1 全部为 rxntype=1
        res1, t1, p1 = evaluate_dataset(df1, "round1", solvent_lookup, qc_lookups)
        all_results.append(res1)
        all_global_trues.append(t1)
        all_global_preds.append(p1)
    else:
        print(f"[跳过] 找不到文件: {round1_path}")

    # ── Round 2 ──
    if round2_path.exists():
        df2 = pd.read_csv(round2_path)
        df2["rxntype"] = pd.to_numeric(df2["rxntype"], errors="coerce").astype(int)
        res2, t2, p2 = evaluate_dataset(df2, "round2", solvent_lookup, qc_lookups)
        all_results.append(res2)
        all_global_trues.append(t2)
        all_global_preds.append(p2)
    else:
        print(f"[跳过] 找不到文件: {round2_path}")

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        print_summary(all_results_df)

        # 全局 pooled R²（round1 + round2 所有样本合并）
        if all_global_trues:
            global_trues = np.concatenate(all_global_trues)
            global_preds = np.concatenate(all_global_preds)
            global_r2 = float(r2_score(global_trues, global_preds))
            global_rmse = float(np.sqrt(mean_squared_error(global_trues, global_preds)))
            print(f"\n  >> 全局 pooled R² (round1+round2) = {global_r2:.6f}  RMSE = {global_rmse:.6f}")

        # 保存结果
        out_path = SCRIPT_DIR.parent / "results" / "layer_rdkit_qc_solvent_test_results.csv"
        all_results_df.to_csv(out_path, index=False)
        print(f"\n结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
