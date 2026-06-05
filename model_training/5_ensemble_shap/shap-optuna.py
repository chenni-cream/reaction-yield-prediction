#!/usr/bin/env python3
"""
shap-optuna.py — Ensemble SHAP Analysis (3-Model Simple Average)

Models:
  A: Avalon FP only                     (ckpt-searchfp/AvalonFingerprint_lgbm)
  B: Avalon FP + Top-500 RDKit          (ckpt-optuna/Avalon_RDKit)
  C: Layered FP + Top-500 NonFP          (ckpt-optuna/Layered_RDKit_QC_Solvent)

Ensemble SHAP = sum of each model's SHAP values × 1/3,
aggregated by readable feature name.
"""
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Patch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import r2_score

try:
    from rdkit.Avalon import pyAvalonTools
except ImportError:
    pyAvalonTools = None

RDLogger.DisableLog("rdApp.*")

# ════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════
MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048
N_RDKIT = 210
N_FOLDS = 5
TOP_K = 500
TOP_DISPLAY = 20

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent.parent / "data"
RDKIT_DIR = DATASET_DIR / "extra-rdkit"

MODEL_A_DIR = SCRIPT_DIR.parent / "ckpt-searchfp" / "AvalonFingerprint_lgbm"
MODEL_B_DIR = SCRIPT_DIR.parent / "ckpt-optuna" / "Avalon_RDKit"
MODEL_C_DIR = SCRIPT_DIR.parent / "ckpt-optuna" / "Layered_RDKit_QC_Solvent"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RXN_NAMES = {
    1: "C\u2013N coupling", 2: "Suzuki", 3: "Heck", 4: "Diels-Alder",
    5: "SNAr", 6: "Sonogashira", 7: "Michael addition", 8: "Amidation",
}

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
N_SOLV_FEATURES = len(ALL_SOLV_COLS)

QC_FEATURE_COLS = [
    "dipole", "electron_affinity", "electrophilicity", "nucleophilicity",
    "electrofugality", "nucleofugality", "homo", "lumo", "homo-lumo",
    "ionization_potential",
]
N_QC_FEATURES = len(QC_FEATURE_COLS)
QC_NAMES_SET = set(QC_FEATURE_COLS)

N_FP = len(MOLECULE_COLUMNS) * FP_SIZE   # 10240
N_RDKIT_TOTAL = len(MOLECULE_COLUMNS) * N_RDKIT  # 1050

# RDKit descriptor names
_DESC_NAMES = pd.read_csv(
    RDKIT_DIR / "train-rdkitfeature-Reactant1_feature_names.csv"
)["FeatureName"].tolist()

# Feature category colors
COLORS = {
    "Avalon FP": "#4393c3",
    "Layered FP": "#762a83",
    "RDKit": "#d6604d",
    "Solvent": "#5aae61",
    "QC": "#f4a582",
}
LEGEND = [
    Patch(facecolor=COLORS["Avalon FP"], label="Avalon FP"),
    Patch(facecolor=COLORS["Layered FP"], label="Layered FP"),
    Patch(facecolor=COLORS["RDKit"], label="RDKit Descriptor"),
    Patch(facecolor=COLORS["Solvent"], label="Solvent Property"),
    Patch(facecolor=COLORS["QC"], label="QC Descriptor"),
]


# ════════════════════════════════════════════════════════════════
# Feature name mapping → readable names
# ════════════════════════════════════════════════════════════════
def map_avalon_name(col, idx):
    return f"Avalon_FP/{col}#{idx}"

def map_layered_name(col, idx):
    return f"Layered_FP/{col}#{idx}"

def map_rdkit_name(col, idx):
    desc = _DESC_NAMES[idx] if idx < len(_DESC_NAMES) else f"desc_{idx}"
    return f"{col}/{desc}"

def map_solv_name(col_name):
    return f"Solvent/{col_name}"

def map_qc_name(col, desc):
    return f"{col}/{desc}"

def get_feature_color(name):
    if "Avalon_FP/" in name:
        return COLORS["Avalon FP"]
    if "Layered_FP/" in name:
        return COLORS["Layered FP"]
    if name.startswith("Solvent/"):
        return COLORS["Solvent"]
    parts = name.split("/", 1)
    if len(parts) == 2 and parts[1] in QC_NAMES_SET:
        return COLORS["QC"]
    return COLORS["RDKit"]

def categorize_feature(name):
    if "Avalon_FP/" in name: return "Avalon FP"
    if "Layered_FP/" in name: return "Layered FP"
    if name.startswith("Solvent/"): return "Solvent"
    parts = name.split("/", 1)
    if len(parts) == 2 and parts[1] in QC_NAMES_SET: return "QC"
    return "RDKit"

def is_fingerprint(name):
    return "Avalon_FP/" in name or "Layered_FP/" in name


# ════════════════════════════════════════════════════════════════
# Full feature name lists (before selection)
# ════════════════════════════════════════════════════════════════
def avalon_feature_names():
    return [map_avalon_name(col, i)
            for col in MOLECULE_COLUMNS for i in range(FP_SIZE)]

def layered_feature_names():
    return [map_layered_name(col, i)
            for col in MOLECULE_COLUMNS for i in range(FP_SIZE)]

def rdkit_feature_names():
    return [map_rdkit_name(col, i)
            for col in MOLECULE_COLUMNS for i in range(N_RDKIT)]

def solvent_feature_names():
    return [map_solv_name(c) for c in ALL_SOLV_COLS]

def qc_feature_names():
    return [map_qc_name(col, d)
            for col in MOLECULE_COLUMNS for d in QC_FEATURE_COLS]


# ════════════════════════════════════════════════════════════════
# Feature building
# ════════════════════════════════════════════════════════════════
def bitvect_to_numpy(bv, n_bits):
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def build_avalon_features(df):
    zero = np.zeros((FP_SIZE,), dtype=np.uint8)
    parts = []
    for col in MOLECULE_COLUMNS:
        rows = []
        for smi in df[col].fillna("").astype(str):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append(zero)
            else:
                rows.append(bitvect_to_numpy(
                    pyAvalonTools.GetAvalonFP(mol, FP_SIZE), FP_SIZE))
        parts.append(np.asarray(rows, dtype=np.float32))
    return np.concatenate(parts, axis=1)

def build_layered_features(df):
    zero = np.zeros((FP_SIZE,), dtype=np.uint8)
    parts = []
    for col in MOLECULE_COLUMNS:
        rows = []
        for smi in df[col].fillna("").astype(str):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append(zero)
            else:
                rows.append(bitvect_to_numpy(
                    Chem.LayeredFingerprint(mol, fpSize=FP_SIZE), FP_SIZE))
        parts.append(np.asarray(rows, dtype=np.float32))
    return np.concatenate(parts, axis=1)

def build_rdkit_lookup(rdkit_dir):
    names_df = pd.read_csv(
        rdkit_dir / "train-rdkitfeature-Reactant1_feature_names.csv")
    desc_names = names_df["FeatureName"].tolist()
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    n_desc = len(desc_names)
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
    for col, fnames in json_priority.items():
        lookup = {}
        for fn in fnames:
            p = rdkit_dir / fn
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for smi, vec in data.items():
                    if smi not in lookup:
                        lookup[smi] = np.where(
                            np.isfinite(vec),
                            np.array(vec, dtype=np.float32), 0.0)
        lookups[col] = (lookup, calc, n_desc)
        print(f"  RDKit '{col}': {len(lookup)} SMILES, {n_desc} desc")
    return lookups

def build_rdkit_features(df, rdkit_lookups):
    parts = []
    for col in MOLECULE_COLUMNS:
        lookup, calc, n_desc = rdkit_lookups[col]
        zero = np.zeros((n_desc,), dtype=np.float32)
        rows = []
        for smi in df[col].fillna("").astype(str):
            if smi in lookup:
                rows.append(lookup[smi])
            else:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    rows.append(zero)
                else:
                    vals = calc.CalcDescriptors(mol)
                    rows.append(np.where(
                        np.isfinite(vals), vals, 0.0).astype(np.float32))
        parts.append(np.asarray(rows, dtype=np.float32))
    return np.concatenate(parts, axis=1)

def build_solvent_lookup(solvents_dir, drugbank_dir):
    main_df = pd.read_csv(solvents_dir / "solvent_withsmiles.csv"
                          ).dropna(subset=["smiles"])
    mnsol_df = pd.read_csv(solvents_dir / "MNSol_alldata_withsmiles.csv"
                           ).dropna(subset=["smiles"])
    drug_df = pd.read_csv(drugbank_dir / "solvent.csv"
                          ).dropna(subset=["smiles"])
    merged = main_df[["smiles"] + SOLV_MAIN_COLS].copy()
    merged = merged.merge(mnsol_df[["smiles"] + MNSOL_COLS],
                          on="smiles", how="outer")
    merged = merged.merge(drug_df[["smiles"] + DRUGBANK_COLS],
                          on="smiles", how="outer")
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
    print(f"  Solvent lookup: {len(lookup)} entries, {N_SOLV_FEATURES} dims")
    return lookup

def build_solvent_features(df, sl):
    zv = np.zeros(N_SOLV_FEATURES, dtype=np.float32)
    cf = []
    for smi in df["Solvent"].fillna("").astype(str):
        vecs = [sl[p.strip()].astype(np.float32)
                for p in smi.split(".") if p.strip() in sl]
        cf.append(np.mean(vecs, axis=0) if vecs else zv)
    return np.asarray(cf, dtype=np.float32)

def build_qc_lookup(qc_dirs):
    file_map = {
        "Reactant1": "psikit_Reactant1.csv.gz",
        "Reactant2": "psikit_Reactant2.csv.gz",
        "Product":   "psikit_Product.csv.gz",
        "Additive":  "psikit_Additive.csv.gz",
        "Solvent":   "psikit_Solvent.csv.gz",
    }
    lookups = {}
    for col, fn in file_map.items():
        lookup = {}
        for d in qc_dirs:
            p = d / fn
            if p.exists():
                with gzip.open(p, "rt") as f:
                    df_qc = pd.read_csv(f)
                for _, row in df_qc.iterrows():
                    smi = row["smile"]
                    if smi not in lookup:
                        lookup[smi] = row[QC_FEATURE_COLS].to_numpy(
                            dtype=np.float32)
        lookups[col] = lookup
        print(f"  QC '{col}': {len(lookup)} SMILES")
    return lookups

def build_qc_features(df, ql):
    zv = np.zeros(N_QC_FEATURES, dtype=np.float32)
    parts = []
    for col in MOLECULE_COLUMNS:
        parts.append(np.asarray(
            [ql[col].get(smi, zv)
             for smi in df[col].fillna("").astype(str)],
            dtype=np.float32))
    return np.concatenate(parts, axis=1)


# ════════════════════════════════════════════════════════════════
# Feature selection (from importance CSV)
# ════════════════════════════════════════════════════════════════
def get_model_b_selected(rxntype):
    """Model B: Avalon FP (all) + Top-K RDKit → (indices, names)."""
    imp_csv = MODEL_B_DIR / f"rxn_{rxntype}" / "feature_importance.csv"
    imp_df = pd.read_csv(imp_csv)
    rdkit_imp = imp_df[imp_df["feature"].str.startswith("RDKit_")]
    rdkit_imp = rdkit_imp[rdkit_imp["importance"] > 0]

    all_names = avalon_feature_names() + rdkit_feature_names()
    indices = [N_FP + MOLECULE_COLUMNS.index(f.split("_", 2)[1]) * N_RDKIT
               + int(f.split("_", 2)[2])
               for f in rdkit_imp["feature"].iloc[:TOP_K]]
    selected = np.concatenate([np.arange(N_FP), np.array(indices)])
    selected.sort()
    return selected, [all_names[i] for i in selected]


def get_model_c_selected(rxntype):
    """Model C: Layered FP (all) + Top-K NonFP → (indices, names)."""
    imp_csv = MODEL_C_DIR / f"rxn_{rxntype}" / "feature_importance.csv"
    imp_df = pd.read_csv(imp_csv)
    nonfp = imp_df[~imp_df["feature"].str.startswith("Layered_")]
    nonfp = nonfp[nonfp["importance"] > 0].reset_index(drop=True)

    all_names = (layered_feature_names() + rdkit_feature_names()
                 + solvent_feature_names() + qc_feature_names())
    indices = []
    for feat in nonfp["feature"].iloc[:TOP_K]:
        if feat.startswith("RDKit_"):
            p = feat.split("_", 2)
            ci = MOLECULE_COLUMNS.index(p[1])
            indices.append(N_FP + ci * N_RDKIT + int(p[2]))
        elif feat.startswith("Solvent_"):
            prop = feat[len("Solvent_"):]
            indices.append(N_FP + N_RDKIT_TOTAL + ALL_SOLV_COLS.index(prop))
        else:  # QC_
            p = feat.split("_", 2)
            ci = MOLECULE_COLUMNS.index(p[1])
            qi = QC_FEATURE_COLS.index(p[2])
            indices.append(N_FP + N_RDKIT_TOTAL + N_SOLV_FEATURES
                           + ci * N_QC_FEATURES + qi)
    selected = np.concatenate([np.arange(N_FP), np.array(indices)])
    selected.sort()
    return selected, [all_names[i] for i in selected]


# ════════════════════════════════════════════════════════════════
# Core: ensemble SHAP for one reaction type
# ════════════════════════════════════════════════════════════════
def run_ensemble_shap(rxntype, rxn_df, rdkit_lookups,
                      solvent_lookup, qc_lookups,
                      fold=1):
    """
    Compute ensemble SHAP for a single reaction type.

    Returns dict of {feature_name: mean_abs_shap} for top features.
    """
    rxn_name = RXN_NAMES[rxntype]
    y_true = rxn_df[TARGET_COLUMN].values.astype(np.float32)
    n = len(rxn_df)
    print(f"\n{'='*60}")
    print(f"  rxntype={rxntype} ({rxn_name}), n={n}, fold={fold}")
    print(f"{'='*60}")

    # Build shared raw features once
    print("  Building features...")
    avalon = build_avalon_features(rxn_df)
    layered = build_layered_features(rxn_df)
    rdkit = build_rdkit_features(rxn_df, rdkit_lookups)
    solv = build_solvent_features(rxn_df, solvent_lookup)
    qc = build_qc_features(rxn_df, qc_lookups)

    # ── Model A: Avalon FP ──
    names_A = avalon_feature_names()
    X_A = avalon
    booster_A = lgb.Booster(
        model_file=str(MODEL_A_DIR / f"rxn_{rxntype}" / f"lgbm_fold{fold}.txt"))
    pred_A = booster_A.predict(X_A)
    r2_A = r2_score(y_true, pred_A)
    print(f"  Model A Fold{fold} R²={r2_A:.6f}")
    sv_A = shap.TreeExplainer(booster_A).shap_values(X_A)
    print(f"  Model A SHAP: {sv_A.shape}")

    # ── Model B: Avalon + Top-500 RDKit ──
    sel_B, names_B = get_model_b_selected(rxntype)
    X_B_full = np.concatenate([avalon, rdkit], axis=1)
    X_B = X_B_full[:, sel_B]
    booster_B = lgb.Booster(
        model_file=str(MODEL_B_DIR / f"rxn_{rxntype}" / f"lgbm_fold{fold}.txt"))
    pred_B = booster_B.predict(X_B)
    r2_B = r2_score(y_true, pred_B)
    print(f"  Model B Fold{fold} R²={r2_B:.6f}")
    sv_B = shap.TreeExplainer(booster_B).shap_values(X_B)
    print(f"  Model B SHAP: {sv_B.shape}")

    # ── Model C: Layered + Top-500 NonFP ──
    sel_C, names_C = get_model_c_selected(rxntype)
    X_C_full = np.concatenate([layered, rdkit, solv, qc], axis=1)
    X_C = X_C_full[:, sel_C]
    booster_C = lgb.Booster(
        model_file=str(MODEL_C_DIR / f"rxn_{rxntype}" / f"lgbm_fold{fold}.txt"))
    pred_C = booster_C.predict(X_C)
    r2_C = r2_score(y_true, pred_C)
    print(f"  Model C Fold{fold} R²={r2_C:.6f}")
    sv_C = shap.TreeExplainer(booster_C).shap_values(X_C)
    print(f"  Model C SHAP: {sv_C.shape}")

    # ── Verify ensemble R² ──
    pred_ens = (pred_A + pred_B + pred_C) / 3.0
    r2_ens = r2_score(y_true, pred_ens)
    print(f"  Ensemble R²={r2_ens:.6f}")

    # ── Aggregate SHAP by feature name (weight=1/3 each model) ──
    feat_shap = defaultdict(lambda: np.zeros(n, dtype=np.float64))
    feat_X = {}  # name → feature values (for beeswarm)

    # Model A: all Avalon FP × 1/3
    for j, name in enumerate(names_A):
        feat_shap[name] += sv_A[:, j] / 3.0
        feat_X[name] = X_A[:, j]

    # Model B: selected features × 1/3
    # Avalon FP features overlap with Model A → SHAP values accumulate
    for j, name in enumerate(names_B):
        feat_shap[name] += sv_B[:, j] / 3.0
        feat_X[name] = X_B[:, j]

    # Model C: selected features × 1/3
    for j, name in enumerate(names_C):
        feat_shap[name] += sv_C[:, j] / 3.0
        feat_X[name] = X_C[:, j]

    # ── Rank by mean |SHAP| (NonFP only) ──
    all_names = list(feat_shap.keys())
    mean_abs = {nm: float(np.abs(feat_shap[nm]).mean()) for nm in all_names}
    nonfp_names = [nm for nm in all_names if not is_fingerprint(nm)]
    sorted_nonfp = sorted(nonfp_names, key=lambda nm: mean_abs[nm], reverse=True)
    top_names = sorted_nonfp[:TOP_DISPLAY]

    print(f"\n  Top-{TOP_DISPLAY} NonFP features (ensemble SHAP):")
    print(f"  {'#':<4} {'Category':<12} {'Feature':<50} {'Mean|SHAP|':<12}")
    print(f"  {'-'*78}")
    for rank, nm in enumerate(top_names, 1):
        cat = categorize_feature(nm)
        print(f"  {rank:<4} {cat:<12} {nm:<50} {mean_abs[nm]:<12.6f}")

    # ── Bar plot ──
    plot_bar(rxntype, rxn_name, n, top_names, mean_abs)

    # ── Beeswarm plot ──
    plot_beeswarm(rxntype, rxn_name, n, top_names,
                  feat_shap, feat_X)

    plt.close("all")
    return {nm: mean_abs[nm] for nm in top_names}


# ════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════
def plot_bar(rxntype, rxn_name, n, top_names, mean_abs):
    top_vals = [mean_abs[nm] for nm in top_names][::-1]
    top_labels = top_names[::-1]
    colors = [get_feature_color(nm) for nm in top_labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_labels)), top_vals, color=colors)
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (Ensemble)")
    ax.set_title(f"SHAP NonFP Feature Importance \u2014 {rxn_name} (Test, n={n})")
    ax.legend(handles=LEGEND, loc="lower right", fontsize=8)
    plt.tight_layout()
    out = RESULTS_DIR / f"shap_ensemble_nonfp_bar_rxn{rxntype}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.name}")


def plot_beeswarm(rxntype, rxn_name, n, top_names, feat_shap, feat_X):
    sv_matrix = np.column_stack([feat_shap[nm] for nm in top_names])
    X_matrix = np.column_stack([feat_X[nm] for nm in top_names])

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_matrix, X_matrix,
                      feature_names=top_names,
                      max_display=TOP_DISPLAY, show=False)
    # 调整字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("SHAP value", fontsize=16, fontweight='bold')
    plt.title(f"{rxn_name} (Test, n={n})", fontsize=16, fontweight='bold')
    # 放大colorbar标签和刻度字体
    fig = plt.gcf()
    for ax in fig.axes:
        if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == "Feature value":
            ax.set_ylabel("Feature value", fontsize=16, fontweight='bold')
            ax.tick_params(labelsize=16)
    plt.tight_layout()
    out = RESULTS_DIR / f"shap_ensemble_nonfp_beeswarm_rxn{rxntype}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"  Saved: {out.name}")


# ════════════════════════════════════════════════════════════════
# Summary across all reactions
# ════════════════════════════════════════════════════════════════
def print_summary(all_results):
    print(f"\n{'='*80}")
    print("SHAP Summary: Top-10 per reaction")
    print(f"{'='*80}")
    print(f"{'Reaction':<20} {'#':<3} {'Category':<12} "
          f"{'Feature':<45} {'Mean|SHAP|':<12}")
    print("-" * 92)
    for rxntype in range(1, 9):
        rn = RXN_NAMES[rxntype]
        feats = all_results[str(rxntype)]
        for i, (feat, val) in enumerate(list(feats.items())[:10]):
            cat = categorize_feature(feat)
            print(f"{rn if i == 0 else '':<20} "
                  f"{i+1:<3} {cat:<12} {feat:<45} {val:<12.6f}")
        print("-" * 92)

    # Category counts
    cat_counter = Counter()
    for rxntype in range(1, 9):
        for feat in all_results[str(rxntype)]:
            cat_counter[categorize_feature(feat)] += 1
    total = sum(cat_counter.values())
    print(f"\nTop-{TOP_DISPLAY} category distribution "
          f"(8 reactions × {TOP_DISPLAY} = {total}):")
    for cat, cnt in cat_counter.most_common():
        print(f"  {cat:<16} {cnt:>4} ({cnt / total * 100:.1f}%)")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    # Load test data
    test_dir = SCRIPT_DIR.parent.parent / "data"
    df_r1 = pd.read_csv(test_dir / "round1_test_data_with_ans.csv")
    df_r1["rxntype"] = 1
    df_r2 = pd.read_csv(test_dir / "round2_test_data_with_ans.csv")
    df_r2["rxntype"] = pd.to_numeric(
        df_r2["rxntype"], errors="coerce").astype(int)
    df_test = pd.concat([df_r1, df_r2], ignore_index=True)
    rxn_groups = {int(k): v.reset_index(drop=True)
                  for k, v in df_test.groupby("rxntype")}
    print(f"Test data: {len(df_test)} samples, "
          f"{len(rxn_groups)} reaction types")

    # Load lookups
    print("\nLoading lookup tables...")
    rdkit_lookups = build_rdkit_lookup(RDKIT_DIR)
    solvent_lookup = build_solvent_lookup(
        DATASET_DIR / "solvents", DATASET_DIR / "drugbank")
    qc_dirs = [DATASET_DIR / d for d in [
        "qm_desc-morfeus-round1", "qm_desc-morfeus",
        "qm_desc-morfeus-round1-test", "qm_desc-morfeus-round2-test",
    ]]
    qc_lookups = build_qc_lookup(qc_dirs)
    print("Lookup tables loaded.\n")

    # Run all 8 reaction types
    all_results = {}
    for rxntype in range(1, 9):
        top = run_ensemble_shap(
            rxntype, rxn_groups[rxntype],
            rdkit_lookups, solvent_lookup, qc_lookups)
        all_results[str(rxntype)] = top

    # Save summary
    summary_path = RESULTS_DIR / "shap_ensemble_nonfp_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
