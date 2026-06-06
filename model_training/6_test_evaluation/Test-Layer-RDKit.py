# Test-Layer-RDKit.py
# Perform inference and evaluation on the test set using the Layered FP + RDKit fusion model
# Load the 5-fold LightGBM model saved in ckpt-layered-rdkit-fusion and predict by grouping by rxntype
# RDKit features are loaded from the pre-generated gz file (run generate_test_rdkit_gz.py first)
import gzip
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048


# ──────────────────────────────────────
# Layered 指纹计算（与训练一致）
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
# RDKit 特征加载（从 gz 文件）
# ──────────────────────────────────────
def load_rdkit_features(rdkit_dir: Path, rxntype: int) -> np.ndarray:
    """加载测试集 RDKit 特征 gz 文件（与训练集格式一致）。"""
    gz_path = rdkit_dir / f"test-rdkitfeature-rxn{rxntype}.gz"
    if not gz_path.exists():
        raise FileNotFoundError(
            f"找不到测试集 RDKit 特征文件: {gz_path}\n"
            f"请先运行: python generate_test_rdkit_gz.py"
        )

    with gzip.open(gz_path, "rb") as f:
        raw = f.read().decode()

    data = []
    for line in raw.strip().split("\n"):
        vals = [float(x) for x in line.split(",")]
        data.append(vals)

    arr = np.asarray(data, dtype=np.float32)

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
# 加载测试数据
# ──────────────────────────────────────
def load_test_data(dataset_dir: Path) -> pd.DataFrame:
    round1_path = dataset_dir / "round1_test_data_with_ans.csv"
    round2_path = dataset_dir / "round2_test_data_with_ans.csv"

    dfs = []
    if round1_path.exists():
        df1 = pd.read_csv(round1_path).copy()
        df1["rxntype"] = 1
        dfs.append(df1)
        print(f"  Round1 测试集: {len(df1)} 样本 (rxntype=1)")
    else:
        print(f"  [警告] 找不到 {round1_path}")

    if round2_path.exists():
        df2 = pd.read_csv(round2_path).copy()
        if "rxntype" not in df2.columns:
            df2["rxntype"] = 2
        dfs.append(df2)
        print(f"  Round2 测试集: {len(df2)} 样本, rxntypes={sorted(df2['rxntype'].unique())}")
    else:
        print(f"  [警告] 找不到 {round2_path}")

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["rxntype"] = pd.to_numeric(df["rxntype"], errors="coerce").astype(int)

    required_cols = MOLECULE_COLUMNS + [TARGET_COLUMN, "rxntype"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"测试集缺少必要列: {missing}")

    return df


# ──────────────────────────────────────
# 加载模型并预测（5 折集成）
# ──────────────────────────────────────
def predict_with_ensemble(
    X: np.ndarray,
    model_dir: Path,
    n_folds: int = 5,
) -> np.ndarray:
    """加载 5 折模型，取平均作为最终预测。"""
    predictions = []
    for fold in range(1, n_folds + 1):
        model_path = model_dir / f"lgbm_fusion_fold{fold}.txt"
        if not model_path.exists():
            print(f"    [警告] 找不到 {model_path}，跳过 fold {fold}")
            continue
        booster = lgb.Booster(model_file=str(model_path))
        pred = booster.predict(X)
        predictions.append(pred)

    if not predictions:
        raise RuntimeError(f"没有可用的模型文件: {model_dir}")

    return np.mean(predictions, axis=0)


# ──────────────────────────────────────
# 主流程
# ──────────────────────────────────────
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent.parent / "data"
    rdkit_dir = dataset_dir / "extra-rdkit"
    ckpt_dir = script_dir.parent / "ckpt-layered-rdkit-fusion" / "fusion"

    print("=" * 70)
    print("Layered FP + RDKit 融合模型 — 测试集推理与评估")
    print("=" * 70)

    # ── 加载测试数据 ──
    print("\n[1/4] 加载测试数据...")
    df = load_test_data(dataset_dir)
    print(f"  总计: {len(df)} 样本")
    rxn_groups = {int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")}
    for rt in sorted(rxn_groups):
        print(f"    rxntype={rt}: {len(rxn_groups[rt])} 样本")

    # ── 逐 rxntype 加载特征 + 预测 ──
    print("\n[2/4] 加载特征并预测...")
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_preds = []
    all_trues = []
    all_rxntypes = []
    all_rxnids = []

    for rxntype in sorted(rxn_groups.keys()):
        rxn_df = rxn_groups[rxntype]
        model_dir = ckpt_dir / f"rxn_{rxntype}"

        if not model_dir.exists():
            print(f"  rxntype={rxntype}: 模型目录不存在 {model_dir}，跳过")
            continue

        start = perf_counter()

        # Layered FP 特征
        layered_feats = build_layered_features(rxn_df)

        # RDKit 特征（从 gz 加载）
        try:
            rdkit_feats = load_rdkit_features(rdkit_dir, rxntype)
        except FileNotFoundError as e:
            print(f"  rxntype={rxntype}: {e}")
            continue

        # 校验样本数
        if rdkit_feats.shape[0] != len(rxn_df):
            print(
                f"  rxntype={rxntype}: [跳过] 样本数不匹配 "
                f"data={len(rxn_df)}, rdkit={rdkit_feats.shape[0]}"
            )
            continue

        # 拼接特征
        X = np.concatenate([layered_feats, rdkit_feats], axis=1)
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        print(
            f"  rxntype={rxntype}: n={len(rxn_df)}, "
            f"Layered={layered_feats.shape[1]}, RDKit={rdkit_feats.shape[1]}, "
            f"总特征={X.shape[1]}"
        )

        # 集成预测
        preds = predict_with_ensemble(X, model_dir, n_folds=5)

        # 评估指标
        r2 = float(r2_score(y, preds))
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))

        elapsed = perf_counter() - start
        print(f"    R2={r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f} ({elapsed:.1f}s)")

        all_preds.extend(preds.tolist())
        all_trues.extend(y.tolist())
        all_rxntypes.extend([rxntype] * len(rxn_df))
        all_rxnids.extend(
            rxn_df["rxnid"].tolist() if "rxnid" in rxn_df.columns else range(len(rxn_df))
        )

        all_rows.append({
            "rxntype": rxntype,
            "n_samples": len(rxn_df),
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "seconds": elapsed,
        })

    if not all_rows:
        print("没有成功预测的 rxntype，退出")
        return

    # ── 总体评估 ──
    print("\n[3/4] 总体评估...")
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    overall_r2 = float(r2_score(all_trues, all_preds))
    overall_rmse = float(np.sqrt(mean_squared_error(all_trues, all_preds)))
    overall_mae = float(mean_absolute_error(all_trues, all_preds))

    print(f"  总体 R2={overall_r2:.6f}, RMSE={overall_rmse:.6f}, MAE={overall_mae:.6f}")

    # ── 保存结果 ──
    print("\n[4/4] 保存结果...")

    pred_df = pd.DataFrame({
        "rxnid": all_rxnids,
        "rxntype": all_rxntypes,
        "true_yield": all_trues,
        "pred_yield": all_preds,
        "abs_error": np.abs(all_trues - all_preds),
    })
    pred_df.to_csv(results_dir / "test_layered_rdkit_predictions.csv", index=False)

    summary_df = pd.DataFrame(all_rows)
    summary_df.loc[len(summary_df)] = {
        "rxntype": "overall",
        "n_samples": len(all_preds),
        "r2": overall_r2,
        "rmse": overall_rmse,
        "mae": overall_mae,
        "seconds": sum(r["seconds"] for r in all_rows),
    }
    summary_df.to_csv(results_dir / "test_layered_rdkit_summary.csv", index=False)

    # 打印汇总表
    print(f"\n{'=' * 80}")
    print("测试集评估汇总")
    print(f"{'=' * 80}")
    print(f"{'rxntype':<10} {'n':<8} {'R2':<14} {'RMSE':<14} {'MAE':<14}")
    print("-" * 60)
    for row in all_rows:
        print(
            f"{row['rxntype']:<10} {row['n_samples']:<8} "
            f"{row['r2']:<14.6f} {row['rmse']:<14.6f} {row['mae']:<14.6f}"
        )
    print("-" * 60)
    print(
        f"{'总体':<10} {len(all_preds):<8} "
        f"{overall_r2:<14.6f} {overall_rmse:<14.6f} {overall_mae:<14.6f}"
    )

    print(f"\n预测详情: {results_dir / 'test_layered_rdkit_predictions.csv'}")
    print(f"评估汇总: {results_dir / 'test_layered_rdkit_summary.csv'}")


if __name__ == "__main__":
    main()
