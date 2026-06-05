# LightGBM: 纯描述符基线（RDKit + 溶剂 + QC，无指纹）
# 按 rxntype 分类，五折交叉验证
# 溶剂描述符来自 solvents/、drugbank/、MNSol 三源合并（去重 31 维）
# QC 描述符来自 qm_desc-morfeus（round2）和 qm_desc-morfeus-round1（每分子列 10 维，共 50 维）
# 混合溶剂按 '.' 拆分后分别查表取平均
import gzip
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"

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


# ──────────────────────────────────────
# Extra-RDKit 特征加载（从 gz 文件）
# ──────────────────────────────────────
def load_rdkit_features(rdkit_dir: Path, rxntype: int) -> np.ndarray:
    gz_path = rdkit_dir / f"train-rdkitfeature-rxn{rxntype}.gz"
    if not gz_path.exists():
        raise FileNotFoundError(f"找不到 RDKit 特征文件: {gz_path}")

    with gzip.open(gz_path, "rb") as f:
        raw = f.read().decode()

    data = []
    for line in raw.strip().split("\n"):
        vals = [float(x) for x in line.split(",")]
        data.append(vals)

    arr = np.asarray(data, dtype=np.float32)

    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    problem_mask = nan_mask | inf_mask
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
def build_solvent_lookup(
    solvents_dir: Path, drugbank_dir: Path
) -> dict[str, np.ndarray]:
    """合并 3 个溶剂源，构建 SMILES → 溶剂特征向量查找表（31 维）"""
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
    lookup: dict[str, np.ndarray] = {}
    for _, row in merged.iterrows():
        smi = row["smiles"]
        if pd.isna(smi):
            continue
        lookup[str(smi)] = row[all_feat_cols].to_numpy(dtype=np.float64)

    # 缺失值用列均值填充
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


def build_solvent_features(
    df: pd.DataFrame, solvent_lookup: dict[str, np.ndarray]
) -> np.ndarray:
    """混合溶剂按 '.' 拆分查表取平均，仅对 Solvent 列"""
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

    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    problem_mask = nan_mask | inf_mask
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
def load_qc_gz(gz_path: Path) -> pd.DataFrame:
    with gzip.open(gz_path, "rt") as f:
        return pd.read_csv(f)


def build_qc_lookup(
    qc_dir_round1: Path, qc_dir_round2: Path
) -> dict[str, dict[str, np.ndarray]]:
    """为每个分子列构建 SMILES → QC 特征向量查找表"""
    file_map = {
        "Reactant1": "psikit_Reactant1.csv.gz",
        "Reactant2": "psikit_Reactant2.csv.gz",
        "Product": "psikit_Product.csv.gz",
        "Additive": "psikit_Additive.csv.gz",
        "Solvent": "psikit_Solvent.csv.gz",
    }

    lookups: dict[str, dict[str, np.ndarray]] = {}

    for col, filename in file_map.items():
        lookup: dict[str, np.ndarray] = {}

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


def build_qc_features(
    df: pd.DataFrame, qc_lookups: dict[str, dict[str, np.ndarray]]
) -> np.ndarray:
    """每个分子列 10 个 QC 特征，共 50 维"""
    parts = []
    zero_vec = np.zeros(N_QC_FEATURES, dtype=np.float32)

    for col in MOLECULE_COLUMNS:
        lookup = qc_lookups[col]
        col_feats = []
        for smi in df[col].fillna("").astype(str):
            col_feats.append(lookup.get(smi, zero_vec))
        parts.append(np.asarray(col_feats, dtype=np.float32))

    arr = np.concatenate(parts, axis=1)

    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    problem_mask = nan_mask | inf_mask
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
def load_train_data(dataset_dir: Path) -> pd.DataFrame:
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

    required_cols = MOLECULE_COLUMNS + [TARGET_COLUMN, "rxntype"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"训练集缺少必要列: {missing}")

    df["rxntype"] = pd.to_numeric(df["rxntype"], errors="coerce")
    if df["rxntype"].isna().any():
        raise ValueError("rxntype 列包含无法解析为数字的值")
    df["rxntype"] = df["rxntype"].astype(int)

    return df


# ──────────────────────────────────────
# LightGBM 参数
# ──────────────────────────────────────
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
# 训练与评估
# ──────────────────────────────────────
def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    rxn_type_value: int,
    output_dir: Path,
    n_splits: int,
    random_state: int,
    start_time: float,
) -> tuple[list[dict], dict]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split_indices = list(kf.split(X))

    fold_iter = enumerate(split_indices, start=1)
    if HAS_TQDM:
        fold_iter = enumerate(
            tqdm(split_indices, total=n_splits, desc=f"rxn_{rxn_type_value} fusion", leave=False),
            start=1,
        )

    rxn_dir = output_dir / f"rxn_{rxn_type_value}"
    rxn_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for fold, (train_idx, val_idx) in fold_iter:
        fold_start = perf_counter()

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=0),
            ],
        )

        preds = model.predict(X_val)
        r2 = float(r2_score(y_val, preds))
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))

        model.booster_.save_model(str(rxn_dir / f"lgbm_fusion_fold{fold}.txt"))

        fold_rows.append(
            {
                "rxntype": int(rxn_type_value),
                "fold": int(fold),
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "best_iteration": int(model.best_iteration_),
                "fold_seconds": float(perf_counter() - fold_start),
            }
        )

    r2_vals = np.array([x["r2"] for x in fold_rows])
    rmse_vals = np.array([x["rmse"] for x in fold_rows])
    mae_vals = np.array([x["mae"] for x in fold_rows])
    sec_vals = np.array([x["fold_seconds"] for x in fold_rows])

    summary = {
        "rxntype": int(rxn_type_value),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "r2_mean": float(np.mean(r2_vals)),
        "r2_sd": float(np.std(r2_vals, ddof=1)),
        "r2_mean_pm_sd": f"{np.mean(r2_vals):.6f} ± {np.std(r2_vals, ddof=1):.6f}",
        "rmse_mean": float(np.mean(rmse_vals)),
        "rmse_sd": float(np.std(rmse_vals, ddof=1)),
        "mae_mean": float(np.mean(mae_vals)),
        "mae_sd": float(np.std(mae_vals, ddof=1)),
        "fold_seconds_mean": float(np.mean(sec_vals)),
        "total_seconds": float(perf_counter() - start_time),
    }
    return fold_rows, summary


# ──────────────────────────────────────
# 特征重要性分析
# ──────────────────────────────────────
def analyze_importance(
    rdkit_feats: np.ndarray,
    solv_feats: np.ndarray,
    qc_feats: np.ndarray,
    y: np.ndarray,
    rxn_type_value: int,
    output_dir: Path,
    top_k: int = 30,
) -> None:
    X = np.concatenate([rdkit_feats, solv_feats, qc_feats], axis=1)

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

    importances = model.feature_importances_

    feature_names = []
    for col in MOLECULE_COLUMNS:
        feature_names.extend([f"RDKit_{col}_{i}" for i in range(210)])
    feature_names.extend([f"Solvent_{feat}" for feat in SOLV_MAIN_COLS + MNSOL_COLS + DRUGBANK_COLS])
    for col in MOLECULE_COLUMNS:
        feature_names.extend([f"QC_{col}_{feat}" for feat in QC_FEATURE_COLS])

    n_rdkit = rdkit_feats.shape[1]
    n_solv = solv_feats.shape[1]
    n_qc = qc_feats.shape[1]
    types = ["RDKit"] * n_rdkit + ["Solvent"] * n_solv + ["QC"] * n_qc
    components = (
        [col for col in MOLECULE_COLUMNS for _ in range(210)]
        + ["Solvent"] * n_solv
        + [col for col in MOLECULE_COLUMNS for _ in range(N_QC_FEATURES)]
    )

    imp_df = (
        pd.DataFrame(
            {
                "feature": feature_names[: len(importances)],
                "importance": importances,
                "type": types[: len(importances)],
                "component": components[: len(importances)],
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    rxn_dir = output_dir / f"rxn_{rxn_type_value}"
    rxn_dir.mkdir(parents=True, exist_ok=True)
    imp_df.head(top_k).to_csv(rxn_dir / "top_features.csv", index=False)

    type_imp = imp_df.groupby("type")["importance"].sum()
    total_imp = type_imp.sum()
    parts = []
    for t in ["RDKit", "Solvent", "QC"]:
        if t in type_imp.index:
            parts.append(f"{t}={type_imp[t] / total_imp * 100:.1f}%")
    print(f"    特征重要性占比: {', '.join(parts)}")

    top10 = imp_df.head(10)
    for t in ["RDKit", "Solvent", "QC"]:
        count = (top10["type"] == t).sum()
        if count > 0:
            print(f"    Top-10 特征中 {t} 占 {count} 个")


# ──────────────────────────────────────
# 覆盖率统计
# ──────────────────────────────────────
def check_solvent_coverage(
    df: pd.DataFrame, solvent_lookup: dict[str, np.ndarray]
) -> None:
    print("\n溶剂描述符覆盖率 (拆分混合溶剂后):")
    total = len(df)
    all_parts = 0
    found_parts = 0
    full_match = 0
    for smi in df["Solvent"].fillna("").astype(str):
        parts = smi.split(".")
        parts_found = sum(1 for p in parts if p.strip() in solvent_lookup)
        all_parts += len(parts)
        found_parts += parts_found
        if parts_found == len(parts):
            full_match += 1

    print(f"  样本总数: {total}")
    print(f"  完全匹配（所有组分都找到）: {full_match}/{total} ({full_match/total*100:.1f}%)")
    print(f"  组分覆盖率: {found_parts}/{all_parts} ({found_parts/all_parts*100:.1f}%)")


def check_qc_coverage(
    df: pd.DataFrame, qc_lookups: dict[str, dict[str, np.ndarray]]
) -> None:
    print("\nQC 描述符覆盖率:")
    for col in MOLECULE_COLUMNS:
        lookup = qc_lookups[col]
        smiles_list = df[col].fillna("").astype(str).tolist()
        found = sum(1 for smi in smiles_list if smi in lookup)
        total = len(smiles_list)
        pct = found / total * 100 if total > 0 else 0
        print(f"  {col}: {found}/{total} ({pct:.1f}%)")


# ──────────────────────────────────────
# 主流程
# ──────────────────────────────────────
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent.parent / "data"
    rdkit_dir = dataset_dir / "extra-rdkit"
    solvents_dir = dataset_dir / "solvents"
    drugbank_dir = dataset_dir / "drugbank"
    qc_dir_round1 = dataset_dir / "qm_desc-morfeus-round1"
    qc_dir_round2 = dataset_dir / "qm_desc-morfeus"
    output_root = script_dir.parent / "ckpt-rdkit-solvent-qc-fusion"
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_train_data(dataset_dir)
    rxn_groups = {int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")}

    print(f"数据总量: {len(df)}, 反应类型数: {len(rxn_groups)}")
    for rt in sorted(rxn_groups):
        print(f"  rxntype={rt}: {len(rxn_groups[rt])} 样本")

    # ── 加载溶剂描述符查找表 ──
    print("\n加载溶剂描述符...")
    solvent_lookup = build_solvent_lookup(solvents_dir, drugbank_dir)
    check_solvent_coverage(df, solvent_lookup)

    # ── 加载 QC 描述符查找表 ──
    print("\n加载 QC 分子描述符...")
    qc_lookups = build_qc_lookup(qc_dir_round1, qc_dir_round2)
    check_qc_coverage(df, qc_lookups)

    # 校验 RDKit 特征文件
    print("\n校验 RDKit 特征文件...")
    for rt in sorted(rxn_groups):
        gz_path = rdkit_dir / f"train-rdkitfeature-rxn{rt}.gz"
        if not gz_path.exists():
            print(f"  [警告] rxntype={rt}: 缺少 {gz_path}")
            continue
        rdkit_arr = load_rdkit_features(rdkit_dir, rt)
        expected = len(rxn_groups[rt])
        actual = rdkit_arr.shape[0]
        status = "OK" if actual == expected else "MISMATCH"
        print(
            f"  rxntype={rt}: 期望 {expected}, 实际 {actual}, 特征数 {rdkit_arr.shape[1]} [{status}]"
        )

    all_start = perf_counter()

    fusion_folds = []
    fusion_summary = []

    print(f"\n{'=' * 70}")
    print("RDKit + Solvent + QC Descriptors 融合训练（无指纹）")
    print(f"{'=' * 70}")

    for rxntype in sorted(rxn_groups.keys()):
        rxn_df = rxn_groups[rxntype]
        print(f"\n  rxntype={rxntype}, n={len(rxn_df)}")

        try:
            rdkit_feats = load_rdkit_features(rdkit_dir, rxntype)
        except FileNotFoundError as e:
            print(f"  [跳过] {e}")
            continue

        if rdkit_feats.shape[0] != len(rxn_df):
            print(f"  [跳过] 样本数不匹配: data={len(rxn_df)}, rdkit={rdkit_feats.shape[0]}")
            continue

        solv_feats = build_solvent_features(rxn_df, solvent_lookup)
        qc_feats = build_qc_features(rxn_df, qc_lookups)
        print(f"    Solvent 特征: {solv_feats.shape}, QC 特征: {qc_feats.shape}")

        X = np.concatenate([rdkit_feats, solv_feats, qc_feats], axis=1)
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        fold_rows, summary_row = run_cv(
            X, y, rxntype, output_root / "fusion", 5, 42, perf_counter(),
        )
        fusion_folds.extend(fold_rows)
        fusion_summary.append(summary_row)

        print(
            f"  rxntype={rxntype} | R2: {summary_row['r2_mean_pm_sd']}"
            f" | RMSE: {summary_row['rmse_mean']:.6f}"
            f" | MAE: {summary_row['mae_mean']:.6f}"
        )

        try:
            analyze_importance(
                rdkit_feats=rdkit_feats,
                solv_feats=solv_feats,
                qc_feats=qc_feats,
                y=y,
                rxn_type_value=rxntype,
                output_dir=output_root / "fusion",
            )
        except Exception as e:
            print(f"    [警告] 特征重要性分析失败: {e}")

    # ── 保存结果 ──
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(fusion_folds).to_csv(
        results_dir / "rdkit_solvent_qc_fusion_fold_metrics.csv", index=False
    )
    pd.DataFrame(fusion_summary).to_csv(
        results_dir / "rdkit_solvent_qc_fusion_summary.csv", index=False
    )

    # 加载纯 RDKit 基线结果对比
    baseline_df = None
    baseline_path = results_dir / "rdkit_baseline_summary.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path).rename(
            columns={
                "r2_mean": "baseline_r2",
                "rmse_mean": "baseline_rmse",
                "mae_mean": "baseline_mae",
            }
        )
        print(f"\n已加载基线结果: {baseline_path}")

    # ── 对比汇总表 ──
    print(f"\n{'=' * 80}")
    print("对比汇总: 纯 RDKit 基线 vs RDKit+Solvent+QC 融合")
    print(f"{'=' * 80}")

    if baseline_df is not None:
        print(
            f"{'rxntype':<10} {'n':<8} "
            f"{'基线 R2':<14} {'融合 R2':<14} {'ΔR2':<12} "
            f"{'基线 RMSE':<14} {'融合 RMSE':<14} {'ΔRMSE':<12}"
        )
        print("-" * 96)

        for fs in fusion_summary:
            rt = fs["rxntype"]
            b = baseline_df[baseline_df["rxntype"] == rt]
            if not b.empty:
                b_r2 = b["baseline_r2"].values[0]
                b_rmse = b["baseline_rmse"].values[0]
                d_r2 = fs["r2_mean"] - b_r2
                d_rmse = fs["rmse_mean"] - b_rmse
                print(
                    f"{rt:<10} {fs['n_samples']:<8} "
                    f"{b_r2:<14.6f} {fs['r2_mean']:<14.6f} {d_r2:+.6f} "
                    f"{b_rmse:<14.6f} {fs['rmse_mean']:<14.6f} {d_rmse:+.6f}"
                )

        base_avg_r2 = baseline_df["baseline_r2"].mean()
        fusion_avg_r2 = np.mean([s["r2_mean"] for s in fusion_summary])
        base_avg_rmse = baseline_df["baseline_rmse"].mean()
        fusion_avg_rmse = np.mean([s["rmse_mean"] for s in fusion_summary])
        print("-" * 96)
        print(
            f"{'平均':<10} {'':8} "
            f"{base_avg_r2:<14.6f} {fusion_avg_r2:<14.6f} {fusion_avg_r2 - base_avg_r2:+.6f} "
            f"{base_avg_rmse:<14.6f} {fusion_avg_rmse:<14.6f} {fusion_avg_rmse - base_avg_rmse:+.6f}"
        )
    else:
        print("未找到基线结果，仅输出融合模型指标:")
        for fs in fusion_summary:
            print(
                f"  rxntype={fs['rxntype']} | R2: {fs['r2_mean_pm_sd']} | RMSE: {fs['rmse_mean']:.6f}"
            )

    print(f"\n模型保存目录: {output_root}")
    print(f"结果保存目录: {results_dir}")
    print(f"总耗时: {perf_counter() - all_start:.2f}s")


if __name__ == "__main__":
    main()
