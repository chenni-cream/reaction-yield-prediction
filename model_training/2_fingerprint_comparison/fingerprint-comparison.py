# 在 round1 + round2 训练集上，按 rxntype 分组比较多种分子指纹 + LGBM 表现。
import warnings
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from rdkit.Avalon import pyAvalonTools

    HAS_AVALON = True
except ImportError:
    HAS_AVALON = False

try:
    from skfp.fingerprints import SECFPFingerprint as SKFPSecfp

    HAS_SKFP_SECFP = True
except ImportError:
    HAS_SKFP_SECFP = False

RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"


def bitvect_to_numpy(bitvect: DataStructs.ExplicitBitVect, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def safe_mol_from_smiles(smi: str) -> Chem.Mol | None:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def morgan_fp(mol: Chem.Mol, radius: int, n_bits: int) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return bitvect_to_numpy(fp, n_bits)


def atom_pair_fp(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    return bitvect_to_numpy(fp, n_bits)


def layered_fp(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    fp = Chem.LayeredFingerprint(mol, fpSize=n_bits)
    return bitvect_to_numpy(fp, n_bits)


def pattern_fp(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    fp = Chem.PatternFingerprint(mol, fpSize=n_bits)
    return bitvect_to_numpy(fp, n_bits)


def rdkit_fp(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
    return bitvect_to_numpy(fp, n_bits)


def torsion_fp(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
    return bitvect_to_numpy(fp, n_bits)


def avalon_fp(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    if not HAS_AVALON:
        raise ImportError("rdkit.Avalon 不可用，无法计算 AvalonFingerprint")
    fp = pyAvalonTools.GetAvalonFP(mol, n_bits)
    return bitvect_to_numpy(fp, n_bits)


def secfp_fallback_fp(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    # 退化方案：用 feature-based Morgan 近似 SECFP。
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=n_bits,
        useFeatures=True,
        useChirality=True,
    )
    return bitvect_to_numpy(fp, n_bits)


def get_fp_config(fp_name: str):
    if fp_name == "AtomPairFingerprint":
        return 2048, lambda mol: atom_pair_fp(mol, 2048)
    if fp_name == "AvalonFingerprint":
        return 2048, lambda mol: avalon_fp(mol, 2048)
    if fp_name == "ECFPFingerprint_rad2":
        return 2048, lambda mol: morgan_fp(mol, radius=2, n_bits=2048)
    if fp_name == "ECFPFingerprint_rad3":
        return 2048, lambda mol: morgan_fp(mol, radius=3, n_bits=2048)
    if fp_name == "ECFPFingerprint_rad4":
        return 2048, lambda mol: morgan_fp(mol, radius=4, n_bits=2048)
    if fp_name == "ECFPFingerprint_3072_rad2":
        return 3072, lambda mol: morgan_fp(mol, radius=2, n_bits=3072)
    if fp_name == "LayeredFingerprint":
        return 2048, lambda mol: layered_fp(mol, 2048)
    if fp_name == "PatternFingerprint":
        return 2048, lambda mol: pattern_fp(mol, 2048)
    if fp_name == "RDKitFingerprint":
        return 2048, lambda mol: rdkit_fp(mol, 2048)
    if fp_name == "SECFPFingerprint":
        return 2048, None
    if fp_name == "TopologicalTorsionFingerprint":
        return 2048, lambda mol: torsion_fp(mol, 2048)
    raise ValueError(f"不支持的指纹: {fp_name}")


def encode_smiles_column(smiles_series: pd.Series, fp_name: str) -> np.ndarray:
    n_bits, encoder = get_fp_config(fp_name)
    smiles = smiles_series.fillna("").astype(str).tolist()
    zero = np.zeros((n_bits,), dtype=np.uint8)

    if fp_name == "SECFPFingerprint" and HAS_SKFP_SECFP:
        secfp = SKFPSecfp(fp_size=n_bits, n_jobs=1)
        rows = []
        bad_count = 0
        for smi in smiles:
            smi = smi.strip()
            if not smi or safe_mol_from_smiles(smi) is None:
                rows.append(zero)
                bad_count += 1
                continue
            try:
                vec = np.asarray(secfp.transform([smi]), dtype=np.float32)
                if vec.ndim == 2 and vec.shape[0] == 1:
                    rows.append(vec[0])
                else:
                    rows.append(np.asarray(vec, dtype=np.float32).reshape(-1)[:n_bits])
            except Exception:
                rows.append(zero)
                bad_count += 1
        if bad_count:
            warnings.warn(
                f"SECFP 编码时跳过了 {bad_count} 条无效或异常 SMILES（以全零向量替代）",
                RuntimeWarning,
            )
        return np.asarray(rows, dtype=np.float32)

    if fp_name == "SECFPFingerprint" and not HAS_SKFP_SECFP:
        warnings.warn(
            "未安装 skfp，SECFPFingerprint 将回退为 feature-based Morgan 近似实现。",
            RuntimeWarning,
        )
        encoder = lambda mol: secfp_fallback_fp(mol, n_bits)

    rows = []
    bad_count = 0
    for smi in smiles:
        smi = smi.strip()
        mol = safe_mol_from_smiles(smi)
        if mol is None:
            rows.append(zero)
            bad_count += 1
        else:
            try:
                rows.append(encoder(mol))
            except Exception:
                rows.append(zero)
                bad_count += 1
    if bad_count:
        warnings.warn(
            f"{fp_name} 编码时跳过了 {bad_count} 条无效或异常 SMILES（以全零向量替代）",
            RuntimeWarning,
        )
    return np.asarray(rows, dtype=np.float32)


def build_features(df: pd.DataFrame, fp_name: str) -> np.ndarray:
    feats = [encode_smiles_column(df[col], fp_name) for col in MOLECULE_COLUMNS]
    return np.concatenate(feats, axis=1)


def evaluate_by_rxntype(
    rxn_df: pd.DataFrame,
    fp_name: str,
    lgb_params: dict,
    rxn_type_value: int,
    fp_output_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[dict], dict]:
    start_time = perf_counter()

    X = build_features(rxn_df, fp_name)
    y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split_indices = list(kf.split(X))

    fold_iter = enumerate(split_indices, start=1)
    if HAS_TQDM:
        fold_iter = enumerate(
            tqdm(split_indices, total=n_splits, desc=f"rxn_{rxn_type_value} folds", leave=False),
            start=1,
        )

    rxn_dir = fp_output_dir / f"rxn_{rxn_type_value}"
    rxn_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for fold, (train_idx, val_idx) in fold_iter:
        fold_start = perf_counter()

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        r2 = float(r2_score(y_val, preds))
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))

        model_file = rxn_dir / f"lgbm_fold{fold}.txt"
        model.booster_.save_model(str(model_file))

        fold_rows.append(
            {
                "fingerprint": fp_name,
                "rxntype": int(rxn_type_value),
                "fold": int(fold),
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "fold_seconds": float(perf_counter() - fold_start),
            }
        )

    r2_values = np.array([x["r2"] for x in fold_rows], dtype=float)
    rmse_values = np.array([x["rmse"] for x in fold_rows], dtype=float)
    mae_values = np.array([x["mae"] for x in fold_rows], dtype=float)
    sec_values = np.array([x["fold_seconds"] for x in fold_rows], dtype=float)

    summary_row = {
        "fingerprint": fp_name,
        "rxntype": int(rxn_type_value),
        "n_samples": int(len(rxn_df)),
        "r2_mean": float(np.mean(r2_values)),
        "r2_sd": float(np.std(r2_values, ddof=1)),
        "r2_mean_pm_sd": f"{np.mean(r2_values):.6f} ± {np.std(r2_values, ddof=1):.6f}",
        "rmse_mean": float(np.mean(rmse_values)),
        "rmse_sd": float(np.std(rmse_values, ddof=1)),
        "mae_mean": float(np.mean(mae_values)),
        "mae_sd": float(np.std(mae_values, ddof=1)),
        "fold_seconds_mean": float(np.mean(sec_values)),
        "fold_seconds_sd": float(np.std(sec_values, ddof=1)),
        "total_seconds": float(perf_counter() - start_time),
    }
    return fold_rows, summary_row


def load_train_data(dataset_dir: Path) -> pd.DataFrame:
    round1_path = dataset_dir / "round1_train_data.csv"
    round2_path = dataset_dir / "round2_train_data.csv"

    if not round1_path.exists():
        raise FileNotFoundError(f"找不到文件: {round1_path}")
    if not round2_path.exists():
        raise FileNotFoundError(f"找不到文件: {round2_path}")

    df1 = pd.read_csv(round1_path).copy()
    df2 = pd.read_csv(round2_path).copy()

    # round1 通常没有 rxntype，默认记为 1；若已有则保留原值。
    if "rxntype" not in df1.columns:
        df1["rxntype"] = 1

    # round2 若已有 rxntype（2-8）则保留；仅在缺失时回退为 2。
    if "rxntype" not in df2.columns:
        df2["rxntype"] = 2

    df = pd.concat([df1, df2], axis=0, ignore_index=True)

    required_cols = MOLECULE_COLUMNS + [TARGET_COLUMN, "rxntype"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"训练集缺少必要列: {missing}")

    # 统一成整数标签，便于后续按 rxntype 分组保存。
    df["rxntype"] = pd.to_numeric(df["rxntype"], errors="coerce")
    if df["rxntype"].isna().any():
        raise ValueError("rxntype 列包含无法解析为数字的值")
    df["rxntype"] = df["rxntype"].astype(int)

    return df


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent.parent / "data"
    output_root = script_dir.parent / "ckpt-searchfp"
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_train_data(dataset_dir)
    rxn_groups = {int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")}

    fingerprint_list = [
        #"AtomPairFingerprint",
        "AvalonFingerprint",
        #"ECFPFingerprint_rad2",
        #"ECFPFingerprint_rad3",
        #"ECFPFingerprint_rad4",
        #"ECFPFingerprint_3072_rad2",
        "LayeredFingerprint",
        #"PatternFingerprint",
        #"RDKitFingerprint",
        #"SECFPFingerprint",
        #"TopologicalTorsionFingerprint",
    ]

    lgb_params = {
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

    all_start = perf_counter()
    fp_iter = fingerprint_list
    if HAS_TQDM:
        fp_iter = tqdm(fingerprint_list, desc="Fingerprint", leave=True)

    for fp_name in fp_iter:
        print(f"\n===== 指纹: {fp_name} =====")

        fp_output_dir = output_root / f"{fp_name}_lgbm"
        fp_output_dir.mkdir(parents=True, exist_ok=True)

        fold_records = []
        summary_records = []

        for rxntype, rxn_df in sorted(rxn_groups.items(), key=lambda x: x[0]):
            print(f"开始训练: rxntype={rxntype}, n={len(rxn_df)}")
            try:
                fold_rows, summary_row = evaluate_by_rxntype(
                    rxn_df=rxn_df,
                    fp_name=fp_name,
                    lgb_params=lgb_params,
                    rxn_type_value=rxntype,
                    fp_output_dir=fp_output_dir,
                    n_splits=5,
                    random_state=42,
                )
                fold_records.extend(fold_rows)
                summary_records.append(summary_row)
                print(
                    f"rxntype={rxntype} | R2 (Mean ± SD): {summary_row['r2_mean_pm_sd']} | "
                    f"RMSE Mean: {summary_row['rmse_mean']:.6f} | "
                    f"MAE Mean: {summary_row['mae_mean']:.6f}"
                )
            except Exception as exc:
                print(f"[跳过] {fp_name} | rxntype={rxntype} 失败: {exc}")

        fold_df = pd.DataFrame(fold_records)
        summary_df = pd.DataFrame(summary_records)

        fold_df.to_csv(fp_output_dir / "cv_fold_metrics.csv", index=False)
        summary_df.to_csv(fp_output_dir / "cv_summary_metrics.csv", index=False)
        print(f"已保存目录: {fp_output_dir}")

    print(f"\n全部训练完成，总耗时: {perf_counter() - all_start:.2f}s")


if __name__ == "__main__":
    main()
