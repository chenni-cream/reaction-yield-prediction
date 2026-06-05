# LightGBM: ECFP4 Fingerprint + Extra-RDKit Descriptors 融合训练
# 按 rxntype 分类，五折交叉验证
# 与纯 ECFP4 FP 基线对比，评估 RDKit 特征带来的增益
import gzip
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
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
# ECFP4 指纹计算 (Morgan radius=2)
# ──────────────────────────────────────
def bitvect_to_numpy(bitvect: DataStructs.ExplicitBitVect, n_bits: int) -> np.ndarray:
	arr = np.zeros((n_bits,), dtype=np.uint8)
	DataStructs.ConvertToNumpyArray(bitvect, arr)
	return arr


def ecfp4_fp(mol: Chem.Mol, n_bits: int) -> np.ndarray:
	fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
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
			rows.append(ecfp4_fp(mol, FP_SIZE))
	return np.asarray(rows, dtype=np.float32)


def build_ecfp4_features(df: pd.DataFrame) -> np.ndarray:
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
# LightGBM 参数（与指纹对比基线一致）
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
# 训练与评估：ECFP4 FP + RDKit 融合
# ──────────────────────────────────────
def evaluate_fusion(
	rxn_df: pd.DataFrame,
	rdkit_feats: np.ndarray,
	rxn_type_value: int,
	output_dir: Path,
	n_splits: int = 5,
	random_state: int = 42,
) -> tuple[list[dict], dict]:
	start_time = perf_counter()

	ecfp4_feats = build_ecfp4_features(rxn_df)
	X = np.concatenate([ecfp4_feats, rdkit_feats], axis=1)
	y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

	print(
		f"    特征维度: ECFP4={ecfp4_feats.shape[1]}, RDKit={rdkit_feats.shape[1]}, "
		f"总={X.shape[1]}"
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
	rxn_type_value: int,
	output_dir: Path,
	top_k: int = 30,
) -> None:
	ecfp4_feats = build_ecfp4_features(rxn_df)
	X = np.concatenate([ecfp4_feats, rdkit_feats], axis=1)
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
		feature_names.extend([f"ECFP4_{col}_{i}" for i in range(FP_SIZE)])
	for col in MOLECULE_COLUMNS:
		feature_names.extend([f"RDKit_{col}_{i}" for i in range(210)])

	n_ecfp4 = ecfp4_feats.shape[1]
	n_rdkit = rdkit_feats.shape[1]
	types = ["ECFP4"] * n_ecfp4 + ["RDKit"] * n_rdkit
	components = (
		[col for col in MOLECULE_COLUMNS for _ in range(FP_SIZE)]
		+ [col for col in MOLECULE_COLUMNS for _ in range(210)]
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
	print(
		f"    特征重要性占比: ECFP4={type_imp.get('ECFP4', 0) / total_imp * 100:.1f}%, "
		f"RDKit={type_imp.get('RDKit', 0) / total_imp * 100:.1f}%"
	)

	top10 = imp_df.head(10)
	rdkit_in_top10 = (top10["type"] == "RDKit").sum()
	print(f"    Top-10 特征中 RDKit 占 {rdkit_in_top10} 个")


# ──────────────────────────────────────
# 主流程
# ──────────────────────────────────────
def main() -> None:
	script_dir = Path(__file__).resolve().parent
	dataset_dir = script_dir.parent.parent / "data"
	rdkit_dir = dataset_dir / "extra-rdkit"
	output_root = script_dir.parent / "ckpt-ecfp4-rdkit-fusion"
	output_root.mkdir(parents=True, exist_ok=True)

	df = load_train_data(dataset_dir)
	rxn_groups = {int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")}

	print(f"数据总量: {len(df)}, 反应类型数: {len(rxn_groups)}")
	for rt in sorted(rxn_groups):
		print(f"  rxntype={rt}: {len(rxn_groups[rt])} 样本")

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
	print("ECFP4 FP + RDKit Descriptors 融合训练")
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

		fold_rows, summary_row = evaluate_fusion(
			rxn_df=rxn_df,
			rdkit_feats=rdkit_feats,
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

		# 特征重要性分析
		try:
			analyze_importance(
				rxn_df=rxn_df,
				rdkit_feats=rdkit_feats,
				rxn_type_value=rxntype,
				output_dir=output_root / "fusion",
			)
		except Exception as e:
			print(f"    [警告] 特征重要性分析失败: {e}")

	# ── 保存结果 ──
	results_dir = script_dir.parent / "results"
	results_dir.mkdir(parents=True, exist_ok=True)

	pd.DataFrame(fusion_folds).to_csv(results_dir / "ecfp4_rdkit_fusion_fold_metrics.csv", index=False)
	pd.DataFrame(fusion_summary).to_csv(results_dir / "ecfp4_rdkit_fusion_summary.csv", index=False)

	# 加载已有的纯 ECFP4 基线结果
	baseline_df = None
	baseline_path = script_dir.parent / "results" / "all_models_summary_metrics.csv"
	if baseline_path.exists():
		baseline_full = pd.read_csv(baseline_path)
		baseline_df = baseline_full[baseline_full["model"] == "LightGBM"][
			["rxntype", "r2_mean", "rmse_mean", "mae_mean"]
		].rename(
			columns={
				"r2_mean": "baseline_r2",
				"rmse_mean": "baseline_rmse",
				"mae_mean": "baseline_mae",
			}
		)
		print(f"\n已加载基线结果: {baseline_path}")

	# ── 对比汇总表 ──
	print(f"\n{'=' * 80}")
	print("对比汇总: 纯 ECFP4 基线 vs ECFP4+RDKit 融合")
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
			print(f"  rxntype={fs['rxntype']} | R2: {fs['r2_mean_pm_sd']} | RMSE: {fs['rmse_mean']:.6f}")

	print(f"\n模型保存目录: {output_root}")
	print(f"结果保存目录: {results_dir}")
	print(f"总耗时: {perf_counter() - all_start:.2f}s")


if __name__ == "__main__":
	main()
