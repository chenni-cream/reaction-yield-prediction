# LayeredFingerprint + RDKit + Solvent + QC ablation experiments
# Train with the full set of LayeredFingerprint + RDKit + Solvent + QC, then select the Top-K non-fingerprint features based on LightGBM importance
# Layered is always retained, and only the RDKit + Solvent + QC (non-fingerprint) part is subjected to ablation
# Classify by rxntype, use five-fold cross-validation, the K value with the best average R2 for 8 rxntypes, and save the training parameters of the optimal model
import gzip
import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048
N_RDKIT = 210

# ──────────────────────────────────────
# 溶剂特征列定义（3 源去重后 31 维）
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
N_SOLV_FEATURES = len(SOLV_MAIN_COLS) + len(MNSOL_COLS) + len(DRUGBANK_COLS)  # 31

# ──────────────────────────────────────
# QC 分子描述符列
# ──────────────────────────────────────
QC_FEATURE_COLS = [
    "dipole", "electron_affinity", "electrophilicity", "nucleophilicity",
    "electrofugality", "nucleofugality", "homo", "lumo", "homo-lumo",
    "ionization_potential",
]
N_QC_FEATURES = len(QC_FEATURE_COLS)  # 10

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
# Layered 指纹
# ──────────────────────────────────────
def bitvect_to_numpy(bitvect, n_bits):
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def layered_fp(mol, n_bits):
    fp = Chem.LayeredFingerprint(mol, fpSize=n_bits)
    return bitvect_to_numpy(fp, n_bits)


def encode_smiles_column(smiles_series):
    smiles = smiles_series.fillna("").astype(str).tolist()
    rows = []
    zero = np.zeros((FP_SIZE,), dtype=np.uint8)
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(zero if mol is None else layered_fp(mol, FP_SIZE))
    return np.asarray(rows, dtype=np.float32)


def build_layered_features(df):
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
# 溶剂描述符加载与查找
# ──────────────────────────────────────
def build_solvent_lookup(solvents_dir, drugbank_dir):
    main_df = pd.read_csv(solvents_dir / "solvent_withsmiles.csv")
    main_df = main_df.dropna(subset=["smiles"])

    mnsol_df = pd.read_csv(solvents_dir / "MNSol_alldata_withsmiles.csv")
    mnsol_df = mnsol_df.dropna(subset=["smiles"])

    drug_df = pd.read_csv(drugbank_dir / "solvent.csv")
    drug_df = drug_df.dropna(subset=["smiles"])

    merged = main_df[["smiles"] + SOLV_MAIN_COLS].copy()
    merged = merged.merge(mnsol_df[["smiles"] + MNSOL_COLS], on="smiles", how="outer")
    merged = merged.merge(drug_df[["smiles"] + DRUGBANK_COLS], on="smiles", how="outer")

    all_feat_cols = SOLV_MAIN_COLS + MNSOL_COLS + DRUGBANK_COLS
    lookup = {}
    for _, row in merged.iterrows():
        smi = row["smiles"]
        if pd.isna(smi):
            continue
        lookup[str(smi)] = row[all_feat_cols].to_numpy(dtype=np.float64)

    arr_all = np.array(list(lookup.values()))
    col_means = np.nanmean(arr_all, axis=0)
    col_means[np.isnan(col_means)] = 0.0

    for smi in lookup:
        vec = lookup[smi]
        nan_mask = np.isnan(vec)
        if nan_mask.any():
            vec[nan_mask] = col_means[nan_mask]
            lookup[smi] = vec

    print(f"  溶剂查找表: {len(lookup)} 条 SMILES, {len(all_feat_cols)} 维特征")
    return lookup


def build_solvent_features(df, solvent_lookup):
    zero_vec = np.zeros(N_SOLV_FEATURES, dtype=np.float32)
    col_feats = []

    for smi in df["Solvent"].fillna("").astype(str):
        parts = smi.split(".")
        vecs = []
        for part in parts:
            part = part.strip()
            if part in solvent_lookup:
                vecs.append(solvent_lookup[part].astype(np.float32))
        if vecs:
            col_feats.append(np.mean(vecs, axis=0))
        else:
            col_feats.append(zero_vec)

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


# ──────────────────────────────────────
# QC 分子描述符加载
# ──────────────────────────────────────
def load_qc_gz(gz_path):
    with gzip.open(gz_path, "rt") as f:
        return pd.read_csv(f)


def build_qc_lookup(qc_dir_round1, qc_dir_round2):
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

        if col != "Product":
            r1_path = qc_dir_round1 / filename
            if r1_path.exists():
                df_r1 = load_qc_gz(r1_path)
                for _, row in df_r1.iterrows():
                    lookup[row["smile"]] = row[QC_FEATURE_COLS].to_numpy(dtype=np.float32)

        r2_path = qc_dir_round2 / filename
        if r2_path.exists():
            df_r2 = load_qc_gz(r2_path)
            for _, row in df_r2.iterrows():
                lookup[row["smile"]] = row[QC_FEATURE_COLS].to_numpy(dtype=np.float32)

        lookups[col] = lookup
        print(f"  QC lookup '{col}': {len(lookup)} unique SMILES")

    return lookups


def build_qc_features(df, qc_lookups):
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


# ──────────────────────────────────────
# 数据加载
# ──────────────────────────────────────
def load_train_data(dataset_dir):
    round1_path = dataset_dir / "round1_train_data.csv"
    round2_path = dataset_dir / "round2_train_data.csv"

    if not round1_path.exists():
        raise FileNotFoundError(f"找不到文件: {round1_path}")
    if not round2_path.exists():
        raise FileNotFoundError(f"找不到文件: {round2_path}")

    df1 = pd.read_csv(round1_path).copy()
    df2 = pd.read_csv(round2_path).copy()

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
    models = []
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
            "best_iteration": int(model.best_iteration_ or LGB_PARAMS["n_estimators"]),
        })
        models.append(model)

    r2_vals = np.array([x["r2"] for x in fold_rows])
    rmse_vals = np.array([x["rmse"] for x in fold_rows])
    mae_vals = np.array([x["mae"] for x in fold_rows])
    summary = {
        "r2_mean": float(np.mean(r2_vals)),
        "r2_sd": float(np.std(r2_vals, ddof=1)),
        "rmse_mean": float(np.mean(rmse_vals)),
        "mae_mean": float(np.mean(mae_vals)),
        "best_iteration_mean": float(np.mean([x["best_iteration"] for x in fold_rows])),
        "best_iteration_sd": float(np.std([x["best_iteration"] for x in fold_rows], ddof=1)),
    }
    return summary, fold_rows, models


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
    solvents_dir = dataset_dir / "solvents"
    drugbank_dir = dataset_dir / "drugbank"
    qc_dir_round1 = dataset_dir / "qm_desc-morfeus-round1"
    qc_dir_round2 = dataset_dir / "qm_desc-morfeus"
    output_root = script_dir.parent / "ckpt-ablation-solvent-qc"
    output_root.mkdir(parents=True, exist_ok=True)
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_train_data(dataset_dir)
    rxn_groups = {int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")}

    print(f"数据总量: {len(df)}, 反应类型数: {len(rxn_groups)}")
    for rt in sorted(rxn_groups):
        print(f"  rxntype={rt}: {len(rxn_groups[rt])} 样本")

    # ── 加载溶剂描述符查找表 ──
    print("\n加载溶剂描述符...")
    solvent_lookup = build_solvent_lookup(solvents_dir, drugbank_dir)

    # ── 加载 QC 描述符查找表 ──
    print("\n加载 QC 分子描述符...")
    qc_lookups = build_qc_lookup(qc_dir_round1, qc_dir_round2)

    # Top-K 候选值（非指纹特征区间）
    k_values = [ 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1050]
    all_results = []
    prepared_data = {}

    all_start = perf_counter()

    # ── Step 1: 全量 LayeredFingerprint+RDKit+Solvent+QC ──
    print(f"\n{'='*70}")
    print("Step 1: LayeredFingerprint + RDKit + Solvent + QC 全量")
    print(f"{'='*70}")

    for rxntype in sorted(rxn_groups.keys()):
        rxn_df = rxn_groups[rxntype]
        try:
            rdkit_arr = load_rdkit_features(rdkit_dir, rxntype)
        except FileNotFoundError:
            continue
        if rdkit_arr.shape[0] != len(rxn_df):
            continue

        layered_feats = build_layered_features(rxn_df)
        solv_feats = build_solvent_features(rxn_df, solvent_lookup)
        qc_feats = build_qc_features(rxn_df, qc_lookups)
        prepared_data[rxntype] = {
            "df": rxn_df,
            "rdkit": rdkit_arr,
            "layered": layered_feats,
            "solv": solv_feats,
            "qc": qc_feats,
        }

        X_full = np.concatenate([layered_feats, rdkit_arr, solv_feats, qc_feats], axis=1)
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        summary, _, _ = train_and_evaluate(X_full, y)
        all_results.append({
            "config": "Layered+RDKit+Solvent+QC_all",
            "k": -1,
            "rxntype": rxntype,
            "n_samples": len(rxn_df),
            "n_features": X_full.shape[1],
            **summary,
        })
        print(f"  rxntype={rxntype} | R2={summary['r2_mean']:.6f} | 特征数={X_full.shape[1]}")

    # ── Step 2: Top-K 非指纹特征消融（RDKit + Solvent + QC） ──
    print(f"\n{'='*70}")
    print("Step 2: LayeredFingerprint + Top-K NonFP(RDKit+Solvent+QC) 消融")
    print(f"{'='*70}")

    for rxntype in sorted(rxn_groups.keys()):
        if rxntype not in prepared_data:
            continue
        rxn_df = prepared_data[rxntype]["df"]
        print(f"\n  rxntype={rxntype}, n={len(rxn_df)}")

        rdkit_arr = prepared_data[rxntype]["rdkit"]
        layered_feats = prepared_data[rxntype]["layered"]
        solv_feats = prepared_data[rxntype]["solv"]
        qc_feats = prepared_data[rxntype]["qc"]

        X_full = np.concatenate([layered_feats, rdkit_arr, solv_feats, qc_feats], axis=1)
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        # 构建特征名
        feature_names = []
        for col in MOLECULE_COLUMNS:
            feature_names.extend([f"Layered_{col}_{i}" for i in range(FP_SIZE)])
        for col in MOLECULE_COLUMNS:
            feature_names.extend([f"RDKit_{col}_{i}" for i in range(N_RDKIT)])
        feature_names.extend([f"Solvent_{feat}" for feat in SOLV_MAIN_COLS + MNSOL_COLS + DRUGBANK_COLS])
        for col in MOLECULE_COLUMNS:
            feature_names.extend([f"QC_{col}_{feat}" for feat in QC_FEATURE_COLS])

        # 获取 importance
        imp_df = get_feature_importance(X_full, y, feature_names)

        imp_output = output_root / f"rxn_{rxntype}"
        imp_output.mkdir(parents=True, exist_ok=True)
        imp_df.to_csv(imp_output / "feature_importance_full.csv", index=False)

        # 提取非指纹特征（RDKit + Solvent + QC）importance 排名
        nonfp_imp = imp_df[~imp_df["feature"].str.startswith("Layered_")].copy()
        nonfp_imp = nonfp_imp[nonfp_imp["importance"] > 0].reset_index(drop=True)
        n_nonzero = len(nonfp_imp)
        print(f"    非指纹非零 importance 特征数: {n_nonzero}")

        # 解析非指纹特征名 -> 全局列索引
        nonfp_global_indices = []
        n_layered = len(MOLECULE_COLUMNS) * FP_SIZE
        n_rdkit_total = len(MOLECULE_COLUMNS) * N_RDKIT
        n_solv = solv_feats.shape[1]
        for _, row in nonfp_imp.iterrows():
            feat = row["feature"]
            if feat.startswith("RDKit_"):
                parts = feat.split("_", 2)
                comp = parts[1]
                local_idx = int(parts[2])
                comp_idx = MOLECULE_COLUMNS.index(comp)
                global_idx = n_layered + comp_idx * N_RDKIT + local_idx
            elif feat.startswith("Solvent_"):
                solv_feat_name = feat[len("Solvent_"):]
                solv_feat_idx = (SOLV_MAIN_COLS + MNSOL_COLS + DRUGBANK_COLS).index(solv_feat_name)
                global_idx = n_layered + n_rdkit_total + solv_feat_idx
            else:  # QC_{molecule}_{descriptor}
                parts = feat.split("_", 2)
                comp = parts[1]
                qc_name = parts[2]
                comp_idx = MOLECULE_COLUMNS.index(comp)
                qc_idx = QC_FEATURE_COLS.index(qc_name)
                global_idx = n_layered + n_rdkit_total + n_solv + comp_idx * N_QC_FEATURES + qc_idx
            nonfp_global_indices.append(global_idx)

        # Layered 固定保留，仅从非指纹特征中做 Top-K
        fixed_indices = np.arange(n_layered)

        rxn_start = perf_counter()
        for k in k_values:
            if k > n_nonzero:
                break

            selected_indices = nonfp_global_indices[:k]
            all_indices = np.concatenate([fixed_indices, np.array(selected_indices)])
            all_indices.sort()
            X_topk = X_full[:, all_indices]

            summary, _, _ = train_and_evaluate(X_topk, y)
            all_results.append({
                "config": f"Layered+NonFP_Top{k}",
                "k": int(k),
                "rxntype": rxntype,
                "n_samples": len(rxn_df),
                "n_features": X_topk.shape[1],
                **summary,
            })

        print(f"    完成 {sum(1 for k in k_values if k <= n_nonzero)} 个 K 值, 耗时 {perf_counter() - rxn_start:.1f}s")

    # ── 汇总 ──
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / "ablation_solvent_qc_topk_results.csv", index=False)

    # 核心目标: 找到一个全局统一 K，使 8 个 rxntype 平均 R2 最优
    print(f"\n{'='*80}")
    print("全局统一 K 值排名 (所有 rxntype 加权平均 R2)")
    print(f"{'='*80}")

    config_avg = results_df.groupby("config").agg(
        simple_r2=("r2_mean", "mean"),
        weighted_r2=("r2_mean", lambda g: np.average(g, weights=results_df.loc[g.index, "n_samples"])),
        simple_rmse=("rmse_mean", "mean"),
        weighted_rmse=("rmse_mean", lambda g: np.average(g, weights=results_df.loc[g.index, "n_samples"])),
        n_features=("n_features", "first"),
    ).sort_values("simple_r2", ascending=False)

    print(f"{'排名':<4} {'配置':<45} {'简单平均R2':<14} {'加权平均R2':<14} {'简单RMSE':<14} {'特征数':<10}")
    print("-" * 111)
    for i, (config, row) in enumerate(config_avg.iterrows(), 1):
        print(f"{i:<4} {config:<45} {row['simple_r2']:<14.6f} {row['weighted_r2']:<14.6f} "
              f"{row['simple_rmse']:<14.6f} {row['n_features']:<10.0f}")

    # 最佳全局配置 (按简单平均 R2 排序，体现通用性)
    best_config = config_avg.index[0]
    best_r2 = config_avg.iloc[0]["simple_r2"]
    print(f"\n>>> 全局最优配置: {best_config}")
    print(f">>> 简单平均 R2: {best_r2:.6f} (通用性主指标)")
    print(f">>> 加权平均 R2: {config_avg.iloc[0]['weighted_r2']:.6f}")

    topk_df = results_df[results_df["k"] > 0].copy()
    n_rxntype_expected = 8
    k_avg = topk_df.groupby("k").agg(
        n_rxntype=("rxntype", "nunique"),
        mean_r2=("r2_mean", "mean"),
        mean_rmse=("rmse_mean", "mean"),
        mean_mae=("mae_mean", "mean"),
    ).sort_values(["mean_r2", "k"], ascending=[False, True])

    valid_k_avg = k_avg[k_avg["n_rxntype"] == n_rxntype_expected].copy()
    if valid_k_avg.empty:
        print("\n[警告] 没有任何 K 在 8 个 rxntype 上全部完成，退化为使用可用 rxntype 最多且平均 R2 最高的 K。")
        fallback = k_avg.sort_values(["n_rxntype", "mean_r2"], ascending=[False, False])
        best_k = int(fallback.index[0])
    else:
        best_k = int(valid_k_avg.index[0])

    print(f"\n>>> 8 个 rxntype 平均 R2 最优 K: {best_k}")
    if best_k in k_avg.index:
        print(f">>> K={best_k} | 覆盖 rxntype 数={int(k_avg.loc[best_k, 'n_rxntype'])} | 平均R2={k_avg.loc[best_k, 'mean_r2']:.6f}")

    k_avg.to_csv(results_dir / "ablation_solvent_qc_k_ranking.csv")

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

    # 保存全局排名
    config_avg.to_csv(results_dir / "ablation_solvent_qc_global_k_ranking.csv")

    best_df = best_sub[["rxntype", "n_samples", "r2_mean", "r2_sd", "rmse_mean", "mae_mean", "n_features"]].copy()
    best_df.to_csv(results_dir / "ablation_solvent_qc_best_global_k_results.csv", index=False)

    # ── 使用最优 K 重新训练并保存训练参数与每折模型 ──
    print(f"\n{'='*80}")
    print(f"最优 K={best_k} 重新训练并保存参数")
    print(f"{'='*80}")
    best_k_rows = []
    best_k_params = {
        "best_k": int(best_k),
        "lgb_params": LGB_PARAMS,
        "k_selection_metric": "mean_r2_across_8_rxntypes",
        "per_rxntype": {},
    }

    for rxntype in sorted(prepared_data.keys()):
        rxn_df = prepared_data[rxntype]["df"]
        rdkit_arr = prepared_data[rxntype]["rdkit"]
        layered_feats = prepared_data[rxntype]["layered"]
        solv_feats = prepared_data[rxntype]["solv"]
        qc_feats = prepared_data[rxntype]["qc"]

        X_full = np.concatenate([layered_feats, rdkit_arr, solv_feats, qc_feats], axis=1)
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        feature_names = []
        for col in MOLECULE_COLUMNS:
            feature_names.extend([f"Layered_{col}_{i}" for i in range(FP_SIZE)])
        for col in MOLECULE_COLUMNS:
            feature_names.extend([f"RDKit_{col}_{i}" for i in range(N_RDKIT)])
        feature_names.extend([f"Solvent_{feat}" for feat in SOLV_MAIN_COLS + MNSOL_COLS + DRUGBANK_COLS])
        for col in MOLECULE_COLUMNS:
            feature_names.extend([f"QC_{col}_{feat}" for feat in QC_FEATURE_COLS])

        imp_df = get_feature_importance(X_full, y, feature_names)
        nonfp_imp = imp_df[~imp_df["feature"].str.startswith("Layered_")].copy()
        nonfp_imp = nonfp_imp[nonfp_imp["importance"] > 0].reset_index(drop=True)
        if nonfp_imp.empty:
            continue

        n_layered = len(MOLECULE_COLUMNS) * FP_SIZE
        n_rdkit_total = len(MOLECULE_COLUMNS) * N_RDKIT
        n_solv = solv_feats.shape[1]
        nonfp_global_indices = []
        for _, row in nonfp_imp.iterrows():
            feat = row["feature"]
            if feat.startswith("RDKit_"):
                parts = feat.split("_", 2)
                comp = parts[1]
                local_idx = int(parts[2])
                comp_idx = MOLECULE_COLUMNS.index(comp)
                global_idx = n_layered + comp_idx * N_RDKIT + local_idx
            elif feat.startswith("Solvent_"):
                solv_feat_name = feat[len("Solvent_"):]
                solv_feat_idx = (SOLV_MAIN_COLS + MNSOL_COLS + DRUGBANK_COLS).index(solv_feat_name)
                global_idx = n_layered + n_rdkit_total + solv_feat_idx
            else:
                parts = feat.split("_", 2)
                comp = parts[1]
                qc_name = parts[2]
                comp_idx = MOLECULE_COLUMNS.index(comp)
                qc_idx = QC_FEATURE_COLS.index(qc_name)
                global_idx = n_layered + n_rdkit_total + n_solv + comp_idx * N_QC_FEATURES + qc_idx
            nonfp_global_indices.append(global_idx)

        fixed_indices = np.arange(n_layered)

        k_use = min(best_k, len(nonfp_global_indices))
        selected_indices = np.array(nonfp_global_indices[:k_use])
        all_indices = np.concatenate([fixed_indices, selected_indices])
        all_indices.sort()
        X_bestk = X_full[:, all_indices]

        summary, fold_rows, models = train_and_evaluate(X_bestk, y)
        best_k_rows.append({
            "rxntype": rxntype,
            "n_samples": len(rxn_df),
            "k": int(k_use),
            "n_features": X_bestk.shape[1],
            **summary,
        })

        model_dir = output_root / f"rxn_{rxntype}" / f"best_k_{k_use}"
        model_dir.mkdir(parents=True, exist_ok=True)
        for fd, model in zip(fold_rows, models):
            model.booster_.save_model(str(model_dir / f"fold{fd['fold']}.txt"))

        best_k_params["per_rxntype"][str(rxntype)] = {
            "n_samples": int(len(rxn_df)),
            "k_used": int(k_use),
            "n_features": int(X_bestk.shape[1]),
            "cv_summary": summary,
            "folds": fold_rows,
        }

    best_k_df = pd.DataFrame(best_k_rows)
    best_k_df.to_csv(results_dir / "ablation_solvent_qc_best_k_cv_results.csv", index=False)
    with open(results_dir / "ablation_solvent_qc_best_k_training_params.json", "w", encoding="utf-8") as f:
        json.dump(best_k_params, f, ensure_ascii=False, indent=2)

    print(f"\n结果保存: {results_dir / 'ablation_solvent_qc_topk_results.csv'}")
    print(f"排名表:   {results_dir / 'ablation_solvent_qc_global_k_ranking.csv'}")
    print(f"最优结果: {results_dir / 'ablation_solvent_qc_best_global_k_results.csv'}")
    print(f"K 排名:   {results_dir / 'ablation_solvent_qc_k_ranking.csv'}")
    print(f"最优K-CV: {results_dir / 'ablation_solvent_qc_best_k_cv_results.csv'}")
    print(f"最优K参数: {results_dir / 'ablation_solvent_qc_best_k_training_params.json'}")
    print(f"总耗时: {perf_counter() - all_start:.1f}s")


if __name__ == "__main__":
    main()
