#!/usr/bin/env python3
"""
ensemble-weighted.py
ensemble-weighted.py
Three-model weighted ensemble prediction (Optuna-tuned version) 
Model A: Avalon FP                          (ckpt-searchfp/AvalonFingerprint_lgbm + OOF from ckpt-optuna/Avalon_FP)
Model B: Avalon FP + Top-500 RDKit           (ckpt-optuna/Avalon_RDKit)
Model C: Layered FP + Top-500 NonFP           (ckpt-optuna/Layered_RDKit_QC_Solvent)

Step 1: Load the previously saved OOF predictions
Step 2: Optimize the integration weights based on rxntype using scipy.optimize
Step 3: Make predictions and evaluate on the test set
"""
import gzip
import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from rdkit.Avalon import pyAvalonTools
except ImportError:
    pyAvalonTools = None

RDLogger.DisableLog("rdApp.*")

# ── 常量 ──────────────────────────────────────────────────────
MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048
N_RDKIT = 210
N_FOLDS = 5

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

QC_FEATURE_COLS = [
    "dipole", "electron_affinity", "electrophilicity", "nucleophilicity",
    "electrofugality", "nucleofugality", "homo", "lumo", "homo-lumo",
    "ionization_potential",
]
N_QC_FEATURES = len(QC_FEATURE_COLS)  # 10

RXN_NAMES = {
    1: "C–N", 2: "Suzuki", 3: "Heck", 4: "Diels-Alder",
    5: "SNAr", 6: "Sonogashira", 7: "Michael", 8: "Amidation",
}


# ══════════════════════════════════════════════════════════════
# 指纹构建
# ══════════════════════════════════════════════════════════════
def bitvect_to_numpy(bitvect, n_bits):
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def avalon_fp(mol, n_bits):
    fp = pyAvalonTools.GetAvalonFP(mol, n_bits)
    return bitvect_to_numpy(fp, n_bits)


def encode_avalon_column(smiles_series):
    smiles = smiles_series.fillna("").astype(str).tolist()
    rows, zero = [], np.zeros((FP_SIZE,), dtype=np.uint8)
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(zero if mol is None else avalon_fp(mol, FP_SIZE))
    return np.asarray(rows, dtype=np.float32)


def build_avalon_features(df):
    return np.concatenate([encode_avalon_column(df[col]) for col in MOLECULE_COLUMNS], axis=1)


def layered_fp(mol, n_bits):
    fp = Chem.LayeredFingerprint(mol, fpSize=n_bits)
    return bitvect_to_numpy(fp, n_bits)


def encode_layered_column(smiles_series):
    smiles = smiles_series.fillna("").astype(str).tolist()
    rows, zero = [], np.zeros((FP_SIZE,), dtype=np.uint8)
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(zero if mol is None else layered_fp(mol, FP_SIZE))
    return np.asarray(rows, dtype=np.float32)


def build_layered_features(df):
    return np.concatenate([encode_layered_column(df[col]) for col in MOLECULE_COLUMNS], axis=1)


# ══════════════════════════════════════════════════════════════
# RDKit 描述符 (JSON 查找 + 实时计算)
# ══════════════════════════════════════════════════════════════
def build_rdkit_lookup(rdkit_dir):
    """从 JSON 文件构建 RDKit 描述符查找表（训练+测试全覆盖）"""
    from rdkit.ML.Descriptors import MoleculeDescriptors

    feature_names_path = rdkit_dir / "train-rdkitfeature-Reactant1_feature_names.csv"
    names_df = pd.read_csv(feature_names_path)
    desc_names = names_df["FeatureName"].tolist()
    n_desc = len(desc_names)
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    json_priority = {
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

    lookups = {}
    for col, filenames in json_priority.items():
        lookup = {}
        for fname in filenames:
            path = rdkit_dir / fname
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                for smi, vec in data.items():
                    if smi not in lookup:
                        arr = np.array(vec, dtype=np.float32)
                        arr = np.where(np.isfinite(arr), arr, 0.0)
                        lookup[smi] = arr
        lookups[col] = (lookup, calc, n_desc)
        print(f"  RDKit lookup '{col}': {len(lookup)} unique SMILES, {n_desc} desc")

    return lookups


def build_rdkit_features(df, rdkit_lookups):
    """用查找表 + 实时计算构建 RDKit 特征"""
    parts = []
    for col in MOLECULE_COLUMNS:
        lookup, calc, n_desc = rdkit_lookups[col]
        zero = np.zeros((n_desc,), dtype=np.float32)
        rows = []
        hit, miss = 0, 0
        for smi in df[col].fillna("").astype(str):
            if smi in lookup:
                rows.append(lookup[smi])
                hit += 1
            else:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    rows.append(zero)
                else:
                    vals = calc.CalcDescriptors(mol)
                    arr = np.array(vals, dtype=np.float32)
                    arr = np.where(np.isfinite(arr), arr, 0.0)
                    rows.append(arr)
                miss += 1
        print(f"    {col}: 查表命中 {hit}, 实时计算 {miss}")
        parts.append(np.asarray(rows, dtype=np.float32))
    return np.concatenate(parts, axis=1)


# ══════════════════════════════════════════════════════════════
# 溶剂描述符
# ══════════════════════════════════════════════════════════════
def build_solvent_lookup(solvents_dir, drugbank_dir):
    main_df = pd.read_csv(solvents_dir / "solvent_withsmiles.csv").dropna(subset=["smiles"])
    mnsol_df = pd.read_csv(solvents_dir / "MNSol_alldata_withsmiles.csv").dropna(subset=["smiles"])
    drug_df = pd.read_csv(drugbank_dir / "solvent.csv").dropna(subset=["smiles"])
    merged = main_df[["smiles"] + SOLV_MAIN_COLS].copy()
    merged = merged.merge(mnsol_df[["smiles"] + MNSOL_COLS], on="smiles", how="outer")
    merged = merged.merge(drug_df[["smiles"] + DRUGBANK_COLS], on="smiles", how="outer")
    lookup = {}
    for _, row in merged.iterrows():
        smi = row["smiles"]
        if pd.isna(smi):
            continue
        lookup[str(smi)] = row[ALL_SOLV_COLS].to_numpy(dtype=np.float64)
    arr_all = np.array(list(lookup.values()))
    col_means = np.nanmean(arr_all, axis=0)
    col_means[np.isnan(col_means)] = 0.0
    for smi in lookup:
        vec = lookup[smi]
        nan_mask = np.isnan(vec)
        if nan_mask.any():
            vec[nan_mask] = col_means[nan_mask]
            lookup[smi] = vec
    print(f"  溶剂查找表: {len(lookup)} 条, {N_SOLV_FEATURES} 维")
    return lookup


def build_solvent_features(df, solvent_lookup):
    zero_vec = np.zeros(N_SOLV_FEATURES, dtype=np.float32)
    col_feats = []
    for smi in df["Solvent"].fillna("").astype(str):
        parts = smi.split(".")
        vecs = [solvent_lookup[p.strip()].astype(np.float32)
                for p in parts if p.strip() in solvent_lookup]
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


# ══════════════════════════════════════════════════════════════
# QC 分子描述符
# ══════════════════════════════════════════════════════════════
def load_qc_gz(gz_path):
    with gzip.open(gz_path, "rt") as f:
        return pd.read_csv(f)


def build_qc_lookup(qc_dirs):
    file_map = {
        "Reactant1": "psikit_Reactant1.csv.gz",
        "Reactant2": "psikit_Reactant2.csv.gz",
        "Product": "psikit_Product.csv.gz",
        "Additive": "psikit_Additive.csv.gz",
        "Solvent": "psikit_Solvent.csv.gz",
    }
    lookups = {}
    for col, filename in file_map.items():
        lookup = {}
        for qc_dir in qc_dirs:
            path = qc_dir / filename
            if path.exists():
                for _, row in load_qc_gz(path).iterrows():
                    smi = row["smile"]
                    if smi not in lookup:
                        lookup[smi] = row[QC_FEATURE_COLS].to_numpy(dtype=np.float32)
        lookups[col] = lookup
        print(f"  QC lookup '{col}': {len(lookup)} unique SMILES")
    return lookups


def build_qc_features(df, qc_lookups):
    zero_vec = np.zeros(N_QC_FEATURES, dtype=np.float32)
    parts = []
    for col in MOLECULE_COLUMNS:
        lookup = qc_lookups[col]
        col_feats = [lookup.get(smi, zero_vec) for smi in df[col].fillna("").astype(str)]
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


# ══════════════════════════════════════════════════════════════
# 特征选择 (从 Optuna 训练保存的 importance CSV 恢复)
# ══════════════════════════════════════════════════════════════
def get_avalon_rdkit_selected_indices(imp_csv_path, top_k=500):
    """配置 B: Avalon + Top-K RDKit 特征选择"""
    imp_df = pd.read_csv(imp_csv_path)
    rdkit_imp = imp_df[imp_df["feature"].str.startswith("RDKit_")]
    rdkit_imp = rdkit_imp[rdkit_imp["importance"] > 0]

    n_avalon = len(MOLECULE_COLUMNS) * FP_SIZE
    indices = []
    for feat in rdkit_imp["feature"].iloc[:top_k]:
        parts = feat.split("_", 2)
        comp, local_idx = parts[1], int(parts[2])
        comp_idx = MOLECULE_COLUMNS.index(comp)
        indices.append(n_avalon + comp_idx * N_RDKIT + local_idx)

    selected = np.concatenate([np.arange(n_avalon), np.array(indices)])
    selected.sort()
    return selected


def get_layered_nonfp_selected_indices(imp_csv_path, n_solv_feats, top_k=500):
    """配置 C: Layered + Top-K 非指纹特征选择"""
    imp_df = pd.read_csv(imp_csv_path)
    nonfp_imp = imp_df[~imp_df["feature"].str.startswith("Layered_")]
    nonfp_imp = nonfp_imp[nonfp_imp["importance"] > 0].reset_index(drop=True)

    n_layered = len(MOLECULE_COLUMNS) * FP_SIZE
    n_rdkit_total = len(MOLECULE_COLUMNS) * N_RDKIT

    indices = []
    for feat in nonfp_imp["feature"].iloc[:top_k]:
        if feat.startswith("RDKit_"):
            parts = feat.split("_", 2)
            comp, local_idx = parts[1], int(parts[2])
            comp_idx = MOLECULE_COLUMNS.index(comp)
            indices.append(n_layered + comp_idx * N_RDKIT + local_idx)
        elif feat.startswith("Solvent_"):
            solv_name = feat[len("Solvent_"):]
            indices.append(n_layered + n_rdkit_total + ALL_SOLV_COLS.index(solv_name))
        else:  # QC_
            parts = feat.split("_", 2)
            comp, qc_name = parts[1], parts[2]
            comp_idx = MOLECULE_COLUMNS.index(comp)
            qc_idx = QC_FEATURE_COLS.index(qc_name)
            indices.append(n_layered + n_rdkit_total + n_solv_feats + comp_idx * N_QC_FEATURES + qc_idx)

    selected = np.concatenate([np.arange(n_layered), np.array(indices)])
    selected.sort()
    return selected


# ══════════════════════════════════════════════════════════════
# 构建各模型测试特征
# ══════════════════════════════════════════════════════════════
def build_test_features_model_A(rxn_df):
    """Model A: Avalon FP only → (n, 10240)"""
    return build_avalon_features(rxn_df)


def build_test_features_model_B(rxn_df, rdkit_lookups, rxntype, imp_root):
    """Model B: Avalon + Top-500 RDKit → (n, ~10740)"""
    avalon = build_avalon_features(rxn_df)
    rdkit = build_rdkit_features(rxn_df, rdkit_lookups)
    X_full = np.concatenate([avalon, rdkit], axis=1)

    imp_csv = imp_root / f"rxn_{rxntype}" / "feature_importance.csv"
    selected = get_avalon_rdkit_selected_indices(imp_csv)
    return X_full[:, selected]


def build_test_features_model_C(rxn_df, rdkit_lookups, rxntype,
                                solv_lookup, qc_lookups, imp_root):
    """Model C: Layered + Top-500 NonFP(RDKit+Solvent+QC) → (n, ~10740)"""
    layered = build_layered_features(rxn_df)
    rdkit = build_rdkit_features(rxn_df, rdkit_lookups)
    solv = build_solvent_features(rxn_df, solv_lookup)
    qc = build_qc_features(rxn_df, qc_lookups)
    X_full = np.concatenate([layered, rdkit, solv, qc], axis=1)

    imp_csv = imp_root / f"rxn_{rxntype}" / "feature_importance.csv"
    selected = get_layered_nonfp_selected_indices(imp_csv, solv.shape[1])
    return X_full[:, selected]


# ══════════════════════════════════════════════════════════════
# 预测工具
# ══════════════════════════════════════════════════════════════
def predict_with_folds(X, model_dir, n_folds=N_FOLDS):
    """n 折模型预测取平均"""
    preds = []
    for i in range(1, n_folds + 1):
        path = model_dir / f"lgbm_fold{i}.txt"
        if path.exists():
            booster = lgb.Booster(model_file=str(path))
            preds.append(booster.predict(X))
    if not preds:
        raise RuntimeError(f"没有可用模型: {model_dir}")
    return np.mean(preds, axis=0)


# ══════════════════════════════════════════════════════════════
# 权重优化
# ══════════════════════════════════════════════════════════════
def optimize_weights(oof_preds_list, y_true):
    """优化集成权重: sum(w)=1, w>=0, 最大化 R²"""
    n_models = len(oof_preds_list)

    def neg_r2(w):
        w = np.array(w)
        w = w / (w.sum() + 1e-12)
        combined = sum(w[i] * oof_preds_list[i] for i in range(n_models))
        return -r2_score(y_true, combined)

    x0 = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(neg_r2, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
    weights = result.x / result.x.sum()
    return weights


# ══════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════
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


def load_test_data(dataset_dir):
    dfs = []
    r1 = dataset_dir / "round1_test_data_with_ans.csv"
    r2 = dataset_dir / "round2_test_data_with_ans.csv"
    if r1.exists():
        df1 = pd.read_csv(r1).copy()
        df1["rxntype"] = 1
        dfs.append(df1)
    if r2.exists():
        df2 = pd.read_csv(r2).copy()
        if "rxntype" not in df2.columns:
            df2["rxntype"] = 2
        dfs.append(df2)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["rxntype"] = pd.to_numeric(df["rxntype"], errors="coerce").astype(int)
    return df


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════
def main():
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent.parent / "data"
    rdkit_dir = dataset_dir / "extra-rdkit"
    solvents_dir = dataset_dir / "solvents"
    drugbank_dir = dataset_dir / "drugbank"
    qc_dirs_train = [
        dataset_dir / "qm_desc-morfeus-round1",
        dataset_dir / "qm_desc-morfeus",
    ]
    qc_dirs_full = qc_dirs_train + [
        dataset_dir / "qm_desc-morfeus-round1-test",
        dataset_dir / "qm_desc-morfeus-round2-test",
    ]

    # ── 模型路径 ──
    ckpt_A = script_dir.parent / "ckpt-optuna" / "Avalon_FP"              # OOF only
    model_A_dir = script_dir.parent / "ckpt-searchfp" / "AvalonFingerprint_lgbm"  # 实际模型
    ckpt_B = script_dir.parent / "ckpt-optuna" / "Avalon_RDKit"           # 模型 + OOF
    ckpt_C = script_dir.parent / "ckpt-optuna" / "Layered_RDKit_QC_Solvent"  # 模型 + OOF

    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    MODEL_NAMES = ["Avalon", "Avalon+RDKit", "Layered+NonFP"]

    # ── Step 0: 加载额外特征查找表 ──
    print("=" * 70)
    print("Step 0: 加载溶剂 & QC & RDKit 查找表")
    print("=" * 70)
    solvent_lookup = build_solvent_lookup(solvents_dir, drugbank_dir)
    rdkit_lookups = build_rdkit_lookup(rdkit_dir)
    qc_lookups_full = build_qc_lookup(qc_dirs_full)

    # ── Step 1: 加载 OOF 预测 & 权重优化 ──
    print(f"\n{'=' * 70}")
    print("Step 1: 加载 OOF 预测 & 权重优化")
    print(f"{'=' * 70}")

    df_train = load_train_data(dataset_dir)
    rxn_groups_train = {int(k): v.reset_index(drop=True)
                        for k, v in df_train.groupby("rxntype")}

    per_rxn_weights = {}

    for rxntype in sorted(rxn_groups_train.keys()):
        rxn_df = rxn_groups_train[rxntype]
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
        rxn_name = RXN_NAMES.get(rxntype, f"rxn_{rxntype}")
        print(f"\n  rxntype={rxntype} ({rxn_name}), n={len(rxn_df)}")

        oof_list = []
        model_labels = []

        # Model A: 加载已保存的 OOF
        oof_path_A = ckpt_A / f"rxn_{rxntype}" / "oof_predictions.npy"
        if oof_path_A.exists():
            oof_A = np.load(str(oof_path_A))
            r2_A = r2_score(y, oof_A)
            oof_list.append(oof_A)
            model_labels.append(MODEL_NAMES[0])
            print(f"    {MODEL_NAMES[0]:20s} OOF R²={r2_A:.6f}")

        # Model B: 加载已保存的 OOF
        oof_path_B = ckpt_B / f"rxn_{rxntype}" / "oof_predictions.npy"
        if oof_path_B.exists():
            oof_B = np.load(str(oof_path_B))
            r2_B = r2_score(y, oof_B)
            oof_list.append(oof_B)
            model_labels.append(MODEL_NAMES[1])
            print(f"    {MODEL_NAMES[1]:20s} OOF R²={r2_B:.6f}")

        # Model C: 加载已保存的 OOF
        oof_path_C = ckpt_C / f"rxn_{rxntype}" / "oof_predictions.npy"
        if oof_path_C.exists():
            oof_C = np.load(str(oof_path_C))
            r2_C = r2_score(y, oof_C)
            oof_list.append(oof_C)
            model_labels.append(MODEL_NAMES[2])
            print(f"    {MODEL_NAMES[2]:20s} OOF R²={r2_C:.6f}")

        if len(oof_list) < 2:
            print(f"    [警告] 可用模型不足 2 个，跳过集成")
            continue

        # 优化权重
        weights = optimize_weights(oof_list, y)
        combined_oof = sum(weights[i] * oof_list[i] for i in range(len(oof_list)))
        r2_combined = r2_score(y, combined_oof)

        per_rxn_weights[rxntype] = {
            "models": model_labels,
            "weights": weights.tolist(),
            "oof_r2_per_model": [float(r2_score(y, oof)) for oof in oof_list],
            "oof_r2_ensemble": float(r2_combined),
        }

        w_str = ", ".join(f"{model_labels[i]}={weights[i]:.4f}" for i in range(len(oof_list)))
        print(f"    集成权重: {w_str}")
        print(f"    集成 OOF R²={r2_combined:.6f}")

    # 保存权重
    weights_path = results_dir / "ensemble_optuna_weights.json"
    with open(weights_path, "w") as f:
        json.dump(per_rxn_weights, f, indent=2, ensure_ascii=False)
    print(f"\n权重已保存: {weights_path}")

    # ── Step 2: 全局权重 ──
    all_oof_by_model = {i: [] for i in range(3)}
    all_y = []
    model_order = [MODEL_NAMES[0], MODEL_NAMES[1], MODEL_NAMES[2]]
    oof_dirs = [ckpt_A, ckpt_B, ckpt_C]

    for rxntype in sorted(rxn_groups_train.keys()):
        rxn_df = rxn_groups_train[rxntype]
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        oof_list = []
        for oof_dir in oof_dirs:
            oof_path = oof_dir / f"rxn_{rxntype}" / "oof_predictions.npy"
            if oof_path.exists():
                oof_list.append(np.load(str(oof_path)))
            else:
                break
        else:
            # 所有三个都有
            for i in range(3):
                all_oof_by_model[i].append(oof_list[i])
            all_y.append(y)

    if all_y:
        all_y_cat = np.concatenate(all_y)
        all_oof_cat = [np.concatenate(all_oof_by_model[i]) for i in range(3)]
        global_weights = optimize_weights(all_oof_cat, all_y_cat)
        global_combined = sum(global_weights[i] * all_oof_cat[i] for i in range(3))
        global_r2 = r2_score(all_y_cat, global_combined)
        w_str = ", ".join(f"{MODEL_NAMES[i]}={global_weights[i]:.4f}" for i in range(3))
        print(f"\n全局权重: {w_str}")
        print(f"全局集成 OOF R²={global_r2:.6f}")
    else:
        global_weights = np.array([1/3, 1/3, 1/3])
        global_r2 = 0.0

    # ── Step 3: 测试集预测 & 评估 ──
    print(f"\n{'=' * 70}")
    print("Step 3: 测试集预测 & 评估")
    print(f"{'=' * 70}")

    df_test = load_test_data(dataset_dir)
    rxn_groups_test = {int(k): v.reset_index(drop=True)
                       for k, v in df_test.groupby("rxntype")}

    eval_rows = []
    all_test_preds_per_rxn = {}
    all_test_true = {}

    for rxntype in sorted(rxn_groups_test.keys()):
        rxn_df = rxn_groups_test[rxntype]
        y_test = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
        rxn_name = RXN_NAMES.get(rxntype, f"rxn_{rxntype}")
        print(f"\n  rxntype={rxntype} ({rxn_name}), n={len(rxn_df)}")

        dir_A = model_A_dir / f"rxn_{rxntype}"
        dir_B = ckpt_B / f"rxn_{rxntype}"
        dir_C = ckpt_C / f"rxn_{rxntype}"

        test_preds = []
        model_labels_test = []

        # Model A: Avalon FP
        if dir_A.exists():
            try:
                X_A = build_test_features_model_A(rxn_df)
                pred_A = predict_with_folds(X_A, dir_A)
                test_preds.append(pred_A)
                model_labels_test.append(MODEL_NAMES[0])
            except Exception as e:
                print(f"  rxn_{rxntype} {MODEL_NAMES[0]} 失败: {e}")

        # Model B: Avalon + RDKit (with feature selection)
        if dir_B.exists():
            try:
                X_B = build_test_features_model_B(rxn_df, rdkit_lookups, rxntype, ckpt_B)
                pred_B = predict_with_folds(X_B, dir_B)
                test_preds.append(pred_B)
                model_labels_test.append(MODEL_NAMES[1])
            except Exception as e:
                print(f"  rxn_{rxntype} {MODEL_NAMES[1]} 失败: {e}")

        # Model C: Layered + NonFP (with feature selection)
        if dir_C.exists():
            try:
                X_C = build_test_features_model_C(rxn_df, rdkit_lookups, rxntype,
                                                   solvent_lookup, qc_lookups_full, ckpt_C)
                pred_C = predict_with_folds(X_C, dir_C)
                test_preds.append(pred_C)
                model_labels_test.append(MODEL_NAMES[2])
            except Exception as e:
                print(f"  rxn_{rxntype} {MODEL_NAMES[2]} 失败: {e}")

        if not test_preds:
            print(f"  rxn_{rxntype} ({rxn_name}): 无可用模型，跳过")
            continue

        # 各单模型测试指标
        single_metrics = {}
        for pred, label in zip(test_preds, model_labels_test):
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            single_metrics[label] = {"r2": r2, "mae": mae, "rmse": rmse}
            print(f"  rxn_{rxntype} ({rxn_name:14s}) {label:20s} Test R²={r2:.6f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

        # 方案 1: Simple Average
        pred_avg = np.mean(test_preds, axis=0)
        r2_avg = r2_score(y_test, pred_avg)
        mae_avg = mean_absolute_error(y_test, pred_avg)
        rmse_avg = np.sqrt(mean_squared_error(y_test, pred_avg))

        # 方案 2: Per-rxn optimized weights
        r2_weighted = -1.0
        mae_weighted = 0.0
        rmse_weighted = 0.0
        pred_weighted = pred_avg.copy()
        if rxntype in per_rxn_weights:
            w_info = per_rxn_weights[rxntype]
            w = np.array(w_info["weights"])
            n_avail = len(w_info["models"])
            if n_avail == len(test_preds):
                pred_weighted = sum(w[i] * test_preds[i] for i in range(n_avail))
            elif n_avail > len(test_preds):
                pred_weighted = sum(w[i] * test_preds[i] for i in range(len(test_preds)))
                pred_weighted /= sum(w[:len(test_preds)])
            r2_weighted = r2_score(y_test, pred_weighted)
            mae_weighted = mean_absolute_error(y_test, pred_weighted)
            rmse_weighted = np.sqrt(mean_squared_error(y_test, pred_weighted))

        # 方案 3: Global weights
        if len(test_preds) == 3:
            pred_global = sum(global_weights[i] * test_preds[i] for i in range(3))
        else:
            pred_global = pred_avg
        r2_global = r2_score(y_test, pred_global)
        mae_global = mean_absolute_error(y_test, pred_global)
        rmse_global = np.sqrt(mean_squared_error(y_test, pred_global))

        # 方案 4: Oracle (best single model)
        best_idx = np.argmax([r2_score(y_test, p) for p in test_preds])
        r2_oracle = r2_score(y_test, test_preds[best_idx])
        mae_oracle = mean_absolute_error(y_test, test_preds[best_idx])
        rmse_oracle = np.sqrt(mean_squared_error(y_test, test_preds[best_idx]))
        oracle_name = model_labels_test[best_idx]

        print(f"  {'':35s} Simple Avg  R²={r2_avg:.6f}  MAE={mae_avg:.4f}  RMSE={rmse_avg:.4f}")
        print(f"  {'':35s} Per-Rxn Wt  R²={r2_weighted:.6f}  MAE={mae_weighted:.4f}  RMSE={rmse_weighted:.4f}")
        print(f"  {'':35s} Global Wt   R²={r2_global:.6f}  MAE={mae_global:.4f}  RMSE={rmse_global:.4f}")
        print(f"  {'':35s} Oracle({oracle_name}) R²={r2_oracle:.6f}  MAE={mae_oracle:.4f}  RMSE={rmse_oracle:.4f}")
        print()

        eval_rows.append({
            "rxntype": rxntype,
            "rxn_name": rxn_name,
            "n_samples": len(rxn_df),
            **{f"test_r2_{lbl.replace('+', '_').replace('-', '_')}": single_metrics[lbl]["r2"]
               for lbl in single_metrics},
            **{f"test_mae_{lbl.replace('+', '_').replace('-', '_')}": single_metrics[lbl]["mae"]
               for lbl in single_metrics},
            **{f"test_rmse_{lbl.replace('+', '_').replace('-', '_')}": single_metrics[lbl]["rmse"]
               for lbl in single_metrics},
            "test_r2_simple_avg": r2_avg,
            "test_r2_per_rxn_weighted": r2_weighted,
            "test_r2_global_weighted": r2_global,
            "test_r2_oracle": r2_oracle,
            "test_mae_simple_avg": mae_avg,
            "test_mae_per_rxn_weighted": mae_weighted,
            "test_mae_global_weighted": mae_global,
            "test_mae_oracle": mae_oracle,
            "test_rmse_simple_avg": rmse_avg,
            "test_rmse_per_rxn_weighted": rmse_weighted,
            "test_rmse_global_weighted": rmse_global,
            "test_rmse_oracle": rmse_oracle,
        })

        all_test_preds_per_rxn[rxntype] = {
            "simple_avg": pred_avg,
            "per_rxn_weighted": pred_weighted,
            "global_weighted": pred_global,
        }
        all_test_true[rxntype] = y_test

    # ── 汇总 ──
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(results_dir / "ensemble_optuna_test_comparison.csv", index=False)

    print(f"\n{'=' * 80}")
    print("测试集评估汇总")
    print(f"{'=' * 80}")

    # 各方法总体指标
    ensemble_methods = [
        ("test_r2_simple_avg", "Simple Avg", "simple_avg"),
        ("test_r2_per_rxn_weighted", "Per-Rxn Wt", "per_rxn_weighted"),
        ("test_r2_global_weighted", "Global Wt", "global_weighted"),
    ]
    overall_rows = []
    for col, name, key in ensemble_methods:
        all_true, all_pred = [], []
        for rxntype in all_test_preds_per_rxn:
            all_true.append(all_test_true[rxntype])
            all_pred.append(all_test_preds_per_rxn[rxntype][key])
        all_true_cat = np.concatenate(all_true)
        all_pred_cat = np.concatenate(all_pred)
        overall_r2 = r2_score(all_true_cat, all_pred_cat)
        overall_mae = mean_absolute_error(all_true_cat, all_pred_cat)
        overall_rmse = np.sqrt(mean_squared_error(all_true_cat, all_pred_cat))
        print(f"  {name:20s} Overall R²={overall_r2:.6f}  MAE={overall_mae:.4f}  RMSE={overall_rmse:.4f}")
        overall_rows.append({
            "method": name,
            "pooled_r2": overall_r2,
            "pooled_mae": overall_mae,
            "pooled_rmse": overall_rmse,
            "n_samples": len(all_true_cat),
        })
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(results_dir / "ensemble_optuna_pooled_metrics.csv", index=False)

    methods = ["test_r2_simple_avg", "test_r2_per_rxn_weighted",
               "test_r2_global_weighted", "test_r2_oracle"]
    method_names = ["Simple Avg", "Per-Rxn Wt", "Global Wt", "Oracle"]

    print(f"\n{'=' * 110}")
    print("逐反应对比 — R²")
    print(f"{'=' * 110}")
    header = f"{'rxn':<14}"
    for name in method_names:
        header += f" {name:<16}"
    print(header)
    print("-" * (14 + 16 * 4))
    for _, row in eval_df.iterrows():
        line = f"{row['rxn_name']:<14}"
        for method in methods:
            line += f" {row[method]:<16.6f}"
        print(line)

    print(f"\n{'=' * 110}")
    print("逐反应对比 — MAE")
    print(f"{'=' * 110}")
    mae_methods = [m.replace("r2", "mae") for m in methods]
    print(header)
    print("-" * (14 + 16 * 4))
    for _, row in eval_df.iterrows():
        line = f"{row['rxn_name']:<14}"
        for method in mae_methods:
            line += f" {row[method]:<16.4f}"
        print(line)

    print(f"\n{'=' * 110}")
    print("逐反应对比 — RMSE")
    print(f"{'=' * 110}")
    rmse_methods = [m.replace("r2", "rmse") for m in methods]
    print(header)
    print("-" * (14 + 16 * 4))
    for _, row in eval_df.iterrows():
        line = f"{row['rxn_name']:<14}"
        for method in rmse_methods:
            line += f" {row[method]:<16.4f}"
        print(line)

    # 保存推荐方案预测
    best_method = eval_df[methods[:-1]].mean().idxmax()
    best_method_name = method_names[methods.index(best_method)]
    print(f"\n>>> 推荐方案: {best_method_name} (平均 R² 最高)")
    print(f">>> 权重文件: {weights_path}")

    save_key = best_method.replace("test_r2_", "")
    all_pred_save, all_true_save, all_rxntype_save = [], [], []
    for rxntype in sorted(all_test_preds_per_rxn.keys()):
        all_pred_save.append(all_test_preds_per_rxn[rxntype][save_key])
        all_true_save.append(all_test_true[rxntype])
        all_rxntype_save.extend([rxntype] * len(all_test_true[rxntype]))

    pred_df = pd.DataFrame({
        "rxntype": all_rxntype_save,
        "true_yield": np.concatenate(all_true_save),
        "pred_yield": np.concatenate(all_pred_save),
    })
    pred_df.to_csv(results_dir / "ensemble_optuna_best_predictions.csv", index=False)

    print(f"\n结果已保存:")
    print(f"  对比表: {results_dir / 'ensemble_optuna_test_comparison.csv'}")
    print(f"  汇总:   {results_dir / 'ensemble_optuna_pooled_metrics.csv'}")
    print(f"  权重:   {results_dir / 'ensemble_optuna_weights.json'}")
    print(f"  预测:   {results_dir / 'ensemble_optuna_best_predictions.csv'}")


if __name__ == "__main__":
    main()
