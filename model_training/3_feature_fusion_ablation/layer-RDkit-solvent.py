# LightGBM: LayeredFingerprint + Extra-RDKit + Solvent Descriptor Fusion Training
# Classified by rxntype, using 5-fold cross-validation
# The solvent descriptors are derived from three sources: solvents/, drugbank/, and MNSol (de-duplicated to 32 dimensions)
# The mixed solvents are split by '.' and the average is taken from the tables respectively
import gzip
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048

# ──────────────────────────────────────
# 溶剂特征列定义（3 源去重后 32 维）
# ──────────────────────────────────────
# solvents/solvent_withsmiles.csv: 去掉与 MNSol 重叠的 Dielectric constant(→eps),
#   去掉与 drugbank 重叠的 HBD(→H_Donor_Count), HBA(→H_Acceptor_Count)
SOLV_MAIN_COLS = [
    "MW (g/mol)", "Density (g/mL)", "Molar volume (mL/mol)", "Refractive index",
    "Mol. refr. pow. (mL/mol)", "Dipole moment (D)", "Melting point (°C)",
    "Boiling point (°C)", "Viscosity (cP)", "lnP (partition coeff.)",
    "Vapour pressure (mbar)", "Henry's constant", "lngamma", "neutral",
]
# solvents/MNSol_alldata_withsmiles.csv
MNSOL_COLS = ["alpha", "beta", "beta**2", "eps", "gamma", "n", "phi**2", "psi**2"]
# drugbank/solvent.csv: solubility_ALOGPS 是字符串，跳过
DRUGBANK_COLS = [
    "logP_ALOGPS", "logS", "logP_ChemAxon", "pKa_acid", "pKa_base",
    "PSA", "Polarizability", "H_Acceptor_Count", "H_Donor_Count",
]
N_SOLV_FEATURES = len(SOLV_MAIN_COLS) + len(MNSOL_COLS) + len(DRUGBANK_COLS)  # 14+8+9=31


# ──────────────────────────────────────
# Layered 指纹计算
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
    """
    构建 SMILES → 溶剂特征向量的查找表。
    合并 3 个来源，去重后共 31 维。
    对同一 SMILES 在多个源中都有数据的做外连接拼接。
    """
    # 1) solvents/solvent_withsmiles.csv
    main_df = pd.read_csv(solvents_dir / "solvent_withsmiles.csv")
    main_df = main_df.dropna(subset=["smiles"])

    # 2) solvents/MNSol_alldata_withsmiles.csv
    mnsol_df = pd.read_csv(solvents_dir / "MNSol_alldata_withsmiles.csv")
    mnsol_df = mnsol_df.dropna(subset=["smiles"])

    # 3) drugbank/solvent.csv
    drug_df = pd.read_csv(drugbank_dir / "solvent.csv")
    drug_df = drug_df.dropna(subset=["smiles"])

    # 合并：按 smiles 做外连接
    merged = main_df[["smiles"] + SOLV_MAIN_COLS].copy()
    merged = merged.merge(mnsol_df[["smiles"] + MNSOL_COLS], on="smiles", how="outer")
    merged = merged.merge(drug_df[["smiles"] + DRUGBANK_COLS], on="smiles", how="outer")

    # 构建查找表
    all_feat_cols = SOLV_MAIN_COLS + MNSOL_COLS + DRUGBANK_COLS
    lookup: dict[str, np.ndarray] = {}
    for _, row in merged.iterrows():
        smi = row["smiles"]
        if pd.isna(smi):
            continue
        vec = row[all_feat_cols].to_numpy(dtype=np.float64)
        # NaN → 用该列的全局均值填充
        lookup[str(smi)] = vec

    # 计算每列均值用于缺失填充
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
    """
    为训练数据构建溶剂特征。仅对 Solvent 列添加。
    混合溶剂（如 'A.B.C'）按 '.' 拆分，分别查表后取平均。
    """
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

    # 处理 NaN / Inf
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
def evaluate_fusion(
    rxn_df: pd.DataFrame,
    rdkit_feats: np.ndarray,
    solv_feats: np.ndarray,
    rxn_type_value: int,
    output_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[dict], dict]:
    start_time = perf_counter()

    layered_feats = build_layered_features(rxn_df)
    X = np.concatenate([layered_feats, rdkit_feats, solv_feats], axis=1)
    y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    print(
        f"    特征维度: Layered={layered_feats.shape[1]}, RDKit={rdkit_feats.shape[1]}, "
        f"Solvent={solv_feats.shape[1]}, 总={X.shape[1]}"
    )

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
        "n_samples": int(len(rxn_df)),
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
    rxn_df: pd.DataFrame,
    rdkit_feats: np.ndarray,
    solv_feats: np.ndarray,
    rxn_type_value: int,
    output_dir: Path,
    top_k: int = 30,
) -> None:
    layered_feats = build_layered_features(rxn_df)
    X = np.concatenate([layered_feats, rdkit_feats, solv_feats], axis=1)
    y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

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
        feature_names.extend([f"Layered_{col}_{i}" for i in range(FP_SIZE)])
    for col in MOLECULE_COLUMNS:
        feature_names.extend([f"RDKit_{col}_{i}" for i in range(210)])
    feature_names.extend([f"Solvent_{feat}" for feat in SOLV_MAIN_COLS + MNSOL_COLS + DRUGBANK_COLS])

    n_layered = layered_feats.shape[1]
    n_rdkit = rdkit_feats.shape[1]
    n_solv = solv_feats.shape[1]
    types = ["Layered"] * n_layered + ["RDKit"] * n_rdkit + ["Solvent"] * n_solv
    components = (
        [col for col in MOLECULE_COLUMNS for _ in range(FP_SIZE)]
        + [col for col in MOLECULE_COLUMNS for _ in range(210)]
        + ["Solvent"] * n_solv
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
    for t in ["Layered", "RDKit", "Solvent"]:
        if t in type_imp.index:
            parts.append(f"{t}={type_imp[t] / total_imp * 100:.1f}%")
    print(f"    特征重要性占比: {', '.join(parts)}")

    top10 = imp_df.head(10)
    for t in ["Layered", "RDKit", "Solvent"]:
        count = (top10["type"] == t).sum()
        if count > 0:
            print(f"    Top-10 特征中 {t} 占 {count} 个")


# ──────────────────────────────────────
# 溶剂覆盖率统计
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


# ──────────────────────────────────────
# 主流程
# ──────────────────────────────────────
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent.parent / "data"
    rdkit_dir = dataset_dir / "extra-rdkit"
    solvents_dir = dataset_dir / "solvents"
    drugbank_dir = dataset_dir / "drugbank"
    output_root = script_dir.parent / "ckpt-layered-rdkit-solvent-fusion"
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_train_data(dataset_dir)
    rxn_groups = {int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")}

    print(f"数据总量: {len(df)}, 反应类型数: {len(rxn_groups)}")
    for rt in sorted(rxn_groups):
        print(f"  rxntype={rt}: {len(rxn_groups[rt])} 样本")

    # ── 加载溶剂描述符查找表 ──
    print("\n加载溶剂描述符...")
    solvent_lookup = build_solvent_lookup(solvents_dir, drugbank_dir)

    # ── 校验覆盖率 ──
    check_solvent_coverage(df, solvent_lookup)

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
    print("Layered FP + RDKit + Solvent Descriptors 融合训练")
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
        print(f"    Solvent 特征矩阵: {solv_feats.shape}")

        fold_rows, summary_row = evaluate_fusion(
            rxn_df=rxn_df,
            rdkit_feats=rdkit_feats,
            solv_feats=solv_feats,
            rxn_type_value=rxntype,
            output_dir=output_root / "fusion",
            n_splits=5,
            random_state=42,
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
                rxn_df=rxn_df,
                rdkit_feats=rdkit_feats,
                solv_feats=solv_feats,
                rxn_type_value=rxntype,
                output_dir=output_root / "fusion",
            )
        except Exception as e:
            print(f"    [警告] 特征重要性分析失败: {e}")

    # ── 保存结果 ──
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(fusion_folds).to_csv(
        results_dir / "layered_rdkit_solvent_fusion_fold_metrics.csv", index=False
    )
    pd.DataFrame(fusion_summary).to_csv(
        results_dir / "layered_rdkit_solvent_fusion_summary.csv", index=False
    )

    # 加载基线结果对比
    baseline_df = None
    baseline_path = results_dir / "layered_rdkit_fusion_summary.csv"
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
    print("对比汇总: Layered+RDKit 基线 vs Layered+RDKit+Solvent 融合")
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
