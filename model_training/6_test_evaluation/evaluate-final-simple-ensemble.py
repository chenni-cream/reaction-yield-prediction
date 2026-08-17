#!/usr/bin/env python3
"""External evaluation of the predefined simple-average ensemble.

The manuscript final model is the arithmetic mean of three component-model
predictions. This script never fits weights or selects a method from test-set
performance; test labels are used only to calculate out-of-sample metrics.
"""
import gzip
import json
from pathlib import Path
from time import perf_counter

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

# ── 常量 ──────────────────────────────────────────────────────
MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048
N_RDKIT = 210
N_FOLDS = 5
MODEL_ORDER = ["Avalon", "Avalon+RDKit", "Layered+NonFP"]

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
    if pyAvalonTools is None:
        raise ImportError("RDKit was installed without Avalon fingerprint support")
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
            # Round1 training data does not contain Product QC descriptors;
            # the file simply does not exist, so path.exists() handles this gracefully.
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
    preds, missing = [], []
    for i in range(1, n_folds + 1):
        path = model_dir / f"lgbm_fold{i}.txt"
        if path.exists():
            booster = lgb.Booster(model_file=str(path))
            preds.append(booster.predict(X))
        else:
            missing.append(path.name)
    if missing:
        raise FileNotFoundError(
            f"Expected {n_folds} fold models in {model_dir}; missing: {missing}"
        )
    return np.mean(preds, axis=0)


# ══════════════════════════════════════════════════════════════
# 测试数据加载
# ══════════════════════════════════════════════════════════════
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
    model_training_dir = script_dir.parent
    dataset_dir = model_training_dir.parent / "data"
    results_dir = model_training_dir / "results"
    rdkit_dir = dataset_dir / "extra-rdkit"
    solvent_lookup = build_solvent_lookup(dataset_dir / "solvents", dataset_dir / "drugbank")
    rdkit_lookups = build_rdkit_lookup(rdkit_dir)
    qc_lookups = build_qc_lookup([
        dataset_dir / "qm_desc-morfeus-round1",
        dataset_dir / "qm_desc-morfeus",
        dataset_dir / "qm_desc-morfeus-round1-test",
        dataset_dir / "qm_desc-morfeus-round2-test",
    ])

    model_dirs = {
        "Avalon": model_training_dir / "ckpt-searchfp" / "AvalonFingerprint_lgbm",
        "Avalon+RDKit": model_training_dir / "ckpt-optuna" / "Avalon_RDKit",
        "Layered+NonFP": model_training_dir / "ckpt-optuna" / "Layered_RDKit_QC_Solvent",
    }
    test_df = load_test_data(dataset_dir)
    metric_rows, all_true, all_pred, prediction_rows = [], [], [], []

    for rxntype, rxn_df in test_df.groupby("rxntype", sort=True):
        rxntype = int(rxntype)
        rxn_df = rxn_df.reset_index(drop=True)
        predictions = {}
        predictions["Avalon"] = predict_with_folds(
            build_test_features_model_A(rxn_df), model_dirs["Avalon"] / f"rxn_{rxntype}"
        )
        predictions["Avalon+RDKit"] = predict_with_folds(
            build_test_features_model_B(rxn_df, rdkit_lookups, rxntype, model_dirs["Avalon+RDKit"]),
            model_dirs["Avalon+RDKit"] / f"rxn_{rxntype}",
        )
        predictions["Layered+NonFP"] = predict_with_folds(
            build_test_features_model_C(
                rxn_df, rdkit_lookups, rxntype, solvent_lookup, qc_lookups,
                model_dirs["Layered+NonFP"],
            ),
            model_dirs["Layered+NonFP"] / f"rxn_{rxntype}",
        )
        pred = np.mean(np.vstack([predictions[name] for name in MODEL_ORDER]), axis=0)
        y_true = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float64)
        r2 = float(r2_score(y_true, pred))
        mae = float(mean_absolute_error(y_true, pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
        metric_rows.append({
            "rxntype": rxntype, "rxn_name": RXN_NAMES.get(rxntype, f"rxn_{rxntype}"),
            "n_samples": len(y_true), "r2": r2, "mae": mae, "rmse": rmse,
        })
        all_true.append(y_true)
        all_pred.append(pred)
        prediction_rows.extend(
            {
                "sample_id": f"rxn_{rxntype}_{sample_index}",
                "rxntype": rxntype,
                "true_yield": true,
                "pred_yield": estimate,
            }
            for sample_index, (true, estimate) in enumerate(zip(y_true, pred))
        )
        print(f"rxn_{rxntype}: n={len(y_true)}, R²={r2:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    true_cat, pred_cat = np.concatenate(all_true), np.concatenate(all_pred)
    pooled = {
        "rxntype": "pooled", "rxn_name": "All", "n_samples": len(true_cat),
        "r2": float(r2_score(true_cat, pred_cat)),
        "mae": float(mean_absolute_error(true_cat, pred_cat)),
        "rmse": float(np.sqrt(mean_squared_error(true_cat, pred_cat))),
    }
    metric_rows.append(pooled)
    metrics_path = results_dir / "final_simple_ensemble_metrics.csv"
    predictions_path = results_dir / "final_simple_ensemble_predictions.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    print(f"\nPooled: n={pooled['n_samples']}, R²={pooled['r2']:.6f}, MAE={pooled['mae']:.6f}, RMSE={pooled['rmse']:.6f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
