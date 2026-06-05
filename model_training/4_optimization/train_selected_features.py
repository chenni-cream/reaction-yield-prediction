# LightGBM: 使用筛选后的特征进行融合训练
# 基于 feature_selection_analysis.py 选出的 327 维特征
# LayeredFingerprint + Extra-RDKit + 溶剂描述符 + 分子级 QC 融合训练
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
# 特征名 → 列索引映射
# ──────────────────────────────────────
def build_feature_name_index() -> dict[str, int]:
    """构建全量特征名 → 列索引的映射表（与原脚本完全一致）"""
    names = []
    # Layered: 5 × 2048 = 10240
    for col in MOLECULE_COLUMNS:
        names.extend([f"Layered_{col}_{i}" for i in range(FP_SIZE)])
    # RDKit: 5 × 210 = 1050
    for col in MOLECULE_COLUMNS:
        names.extend([f"RDKit_{col}_{i}" for i in range(210)])
    # Solvent: 31
    names.extend([f"Solvent_{feat}" for feat in SOLV_MAIN_COLS + MNSOL_COLS + DRUGBANK_COLS])
    # QC: 5 × 10 = 50
    for col in MOLECULE_COLUMNS:
        names.extend([f"QC_{col}_{feat}" for feat in QC_FEATURE_COLS])
    return {name: idx for idx, name in enumerate(names)}


# ──────────────────────────────────────
# 加载筛选特征列表
# ──────────────────────────────────────
def load_selected_features(script_dir: Path) -> tuple[list[str], list[int]]:
    """读取 selected_features_combined.csv，返回特征名列表和对应列索引"""
    csv_path = script_dir / "selected_features_combined.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到筛选特征文件: {csv_path}")

    df = pd.read_csv(csv_path)
    selected_names = df["feature"].tolist()

    name_to_idx = build_feature_name_index()

    valid_names = []
    valid_indices = []
    missing = []
    for name in selected_names:
        if name in name_to_idx:
            valid_names.append(name)
            valid_indices.append(name_to_idx[name])
        else:
            missing.append(name)

    if missing:
        print(f"  [警告] {len(missing)} 个特征名无法映射到列索引，已跳过:")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... 及其他 {len(missing)-10} 个")

    # 按索引排序以保持特征顺序一致性
    sorted_pairs = sorted(zip(valid_indices, valid_names))
    sorted_indices = [p[0] for p in sorted_pairs]
    sorted_names = [p[1] for p in sorted_pairs]

    return sorted_names, sorted_indices


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
# Extra-RDKit 特征加载
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
# 特征筛选：从全量矩阵中提取选中列
# ──────────────────────────────────────
def select_features(
    layered_feats: np.ndarray,
    rdkit_feats: np.ndarray,
    solv_feats: np.ndarray,
    qc_feats: np.ndarray,
    selected_indices: list[int],
    selected_names: list[str],
) -> np.ndarray:
    """拼接全量特征矩阵，然后仅提取筛选后的列"""
    X_full = np.concatenate([layered_feats, rdkit_feats, solv_feats, qc_feats], axis=1)
    X_selected = X_full[:, selected_indices]

    n_layered = layered_feats.shape[1]
    n_rdkit = rdkit_feats.shape[1]
    n_solv = solv_feats.shape[1]
    n_qc = qc_feats.shape[1]
    n_fp_selected = sum(1 for idx in selected_indices if idx < n_layered + n_rdkit)
    n_solv_selected = sum(1 for idx in selected_indices if n_layered + n_rdkit <= idx < n_layered + n_rdkit + n_solv)
    n_qc_selected = sum(1 for idx in selected_indices if idx >= n_layered + n_rdkit + n_solv)

    print(f"    全量特征: {X_full.shape[1]} 维 → 筛选后: {X_selected.shape[1]} 维")
    print(f"    指纹: {n_fp_selected}, 溶剂: {n_solv_selected}, QC: {n_qc_selected}")

    return X_selected


# ──────────────────────────────────────
# 训练与评估
# ──────────────────────────────────────
def evaluate_fusion(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    rxn_type_value: int,
    output_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[dict], dict]:
    start_time = perf_counter()

    print(f"    样本数: {len(y)}, 特征数: {X.shape[1]}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split_indices = list(kf.split(X))

    fold_iter = enumerate(split_indices, start=1)
    if HAS_TQDM:
        fold_iter = enumerate(
            tqdm(split_indices, total=n_splits, desc=f"rxn_{rxn_type_value}", leave=False),
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

        model.booster_.save_model(str(rxn_dir / f"lgbm_selected_fold{fold}.txt"))

        # 保存特征重要性（仅第1折）
        if fold == 1:
            imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            imp_df.to_csv(rxn_dir / "feature_importance.csv", index=False)

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
    print(f"  完全匹配: {full_match}/{total} ({full_match/total*100:.1f}%)")
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
    output_root = script_dir.parent / "ckpt-selected-features"
    output_root.mkdir(parents=True, exist_ok=True)

    # ── 加载筛选特征列表 ──
    print("=" * 70)
    print("加载筛选特征列表...")
    print("=" * 70)
    selected_names, selected_indices = load_selected_features(script_dir)
    print(f"  共 {len(selected_names)} 个筛选特征, 列索引范围: {min(selected_indices)}-{max(selected_indices)}")

    # ── 加载训练数据 ──
    print("\n加载训练数据...")
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

    # ── 校验 RDKit 特征文件 ──
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

    all_folds = []
    all_summary = []

    print(f"\n{'=' * 70}")
    print(f"LightGBM 筛选特征训练 ({len(selected_names)} 维)")
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

        # 构建各部分特征
        layered_feats = build_layered_features(rxn_df)
        solv_feats = build_solvent_features(rxn_df, solvent_lookup)
        qc_feats = build_qc_features(rxn_df, qc_lookups)

        print(f"    Layered={layered_feats.shape[1]}, RDKit={rdkit_feats.shape[1]}, "
              f"Solvent={solv_feats.shape[1]}, QC={qc_feats.shape[1]}")

        # 筛选特征
        X = select_features(
            layered_feats, rdkit_feats, solv_feats, qc_feats,
            selected_indices, selected_names,
        )
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        # 五折训练
        fold_rows, summary_row = evaluate_fusion(
            X=X,
            y=y,
            feature_names=selected_names,
            rxn_type_value=rxntype,
            output_dir=output_root,
            n_splits=5,
            random_state=42,
        )
        all_folds.extend(fold_rows)
        all_summary.append(summary_row)

        print(
            f"  rxntype={rxntype} | R2: {summary_row['r2_mean_pm_sd']}"
            f" | RMSE: {summary_row['rmse_mean']:.6f}"
            f" | MAE: {summary_row['mae_mean']:.6f}"
        )

    # ── 保存结果 ──
    pd.DataFrame(all_folds).to_csv(
        output_root / "fold_metrics.csv", index=False
    )
    pd.DataFrame(all_summary).to_csv(
        output_root / "summary.csv", index=False
    )

    # ── 汇总表 ──
    print(f"\n{'=' * 80}")
    print(f"汇总: LightGBM 筛选特征训练结果 ({len(selected_names)} 维特征)")
    print(f"{'=' * 80}")
    print(
        f"{'rxntype':<10} {'n':<8} {'features':<10} "
        f"{'R2 (mean±sd)':<24} {'RMSE':<14} {'MAE':<14} {'time(s)':<10}"
    )
    print("-" * 90)
    for s in all_summary:
        print(
            f"{s['rxntype']:<10} {s['n_samples']:<8} {s['n_features']:<10} "
            f"{s['r2_mean_pm_sd']:<24} {s['rmse_mean']:<14.6f} {s['mae_mean']:<14.6f} "
            f"{s['total_seconds']:<10.1f}"
        )

    if all_summary:
        avg_r2 = np.mean([s["r2_mean"] for s in all_summary])
        avg_rmse = np.mean([s["rmse_mean"] for s in all_summary])
        avg_mae = np.mean([s["mae_mean"] for s in all_summary])
        print("-" * 90)
        print(
            f"{'平均':<10} {'':8} {'':10} "
            f"{avg_r2:<24.6f} {avg_rmse:<14.6f} {avg_mae:<14.6f}"
        )

    print(f"\n模型保存目录: {output_root}")
    print(f"总耗时: {perf_counter() - all_start:.2f}s")


if __name__ == "__main__":
    main()
