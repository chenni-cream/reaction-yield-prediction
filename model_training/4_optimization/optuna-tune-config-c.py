# Optuna 超参调优: 配置 C 独立脚本
# Layered FP + Top-500 NonFP (RDKit + QC + Solvent)
# 与 optuna-tune-extended.py 并行运行，输出到同一目录
# 只需 Layered 指纹，不需要 Avalon，节省内存
import gzip
import json
import sys
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
TARGET_COLUMN = "Yield"
FP_SIZE = 2048
N_RDKIT = 210
TOP_K = 500
N_FOLDS = 5
N_TRIALS = 50

BASELINE_PARAMS = {
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


# ═══════════════════════════════════════════
# 特征构建
# ═══════════════════════════════════════════
def bitvect_to_numpy(bitvect, n_bits):
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def clean_array(arr):
    problem_mask = np.isnan(arr) | np.isinf(arr)
    if problem_mask.any():
        col_means = np.nanmean(arr, axis=0)
        col_means[np.isnan(col_means)] = 0.0
        for j in range(arr.shape[1]):
            mask = problem_mask[:, j]
            if mask.any():
                arr[mask, j] = col_means[j]
    return arr


def layered_fp(mol, n_bits):
    fp = Chem.LayeredFingerprint(mol, fpSize=n_bits)
    return bitvect_to_numpy(fp, n_bits)


def encode_layered_column(smiles_series):
    smiles = smiles_series.fillna("").astype(str).tolist()
    rows = []
    zero = np.zeros((FP_SIZE,), dtype=np.uint8)
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(zero if mol is None else layered_fp(mol, FP_SIZE))
    return np.asarray(rows, dtype=np.float32)


def build_layered_features(df):
    return np.concatenate([encode_layered_column(df[col]) for col in MOLECULE_COLUMNS], axis=1)


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
    return clean_array(arr)


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

    print(f"  溶剂查找表: {len(lookup)} 条 SMILES, {N_SOLV_FEATURES} 维")
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
        col_feats.append(np.mean(vecs, axis=0) if vecs else zero_vec)
    arr = np.asarray(col_feats, dtype=np.float32)
    return clean_array(arr)


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
            gz_path = qc_dir / filename
            if not gz_path.exists():
                continue
            with gzip.open(gz_path, "rt") as f:
                df_qc = pd.read_csv(f)
            for _, row in df_qc.iterrows():
                smi = row["smile"]
                if smi not in lookup:
                    lookup[smi] = row[QC_FEATURE_COLS].to_numpy(dtype=np.float32)
        lookups[col] = lookup
        print(f"  QC lookup '{col}': {len(lookup)} unique SMILES")
    return lookups


def build_qc_features(df, qc_lookups):
    parts = []
    zero_vec = np.zeros(N_QC_FEATURES, dtype=np.float32)
    for col in MOLECULE_COLUMNS:
        lookup = qc_lookups[col]
        col_feats = [lookup.get(smi, zero_vec) for smi in df[col].fillna("").astype(str)]
        parts.append(np.asarray(col_feats, dtype=np.float32))
    arr = np.concatenate(parts, axis=1)
    return clean_array(arr)


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


# ═══════════════════════════════════════════
# 特征选择
# ═══════════════════════════════════════════
def get_importance(X, y, feature_names):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(kf.split(X))
    model = lgb.LGBMRegressor(**BASELINE_PARAMS)
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


def select_topk_nonfp(X_full, y, solv_feats, top_k=TOP_K):
    feature_names = []
    for col in MOLECULE_COLUMNS:
        feature_names.extend([f"Layered_{col}_{i}" for i in range(FP_SIZE)])
    for col in MOLECULE_COLUMNS:
        feature_names.extend([f"RDKit_{col}_{i}" for i in range(N_RDKIT)])
    feature_names.extend([f"Solvent_{feat}" for feat in ALL_SOLV_COLS])
    for col in MOLECULE_COLUMNS:
        feature_names.extend([f"QC_{col}_{feat}" for feat in QC_FEATURE_COLS])

    imp_df = get_importance(X_full, y, feature_names)

    nonfp_imp = imp_df[~imp_df["feature"].str.startswith("Layered_")]
    nonfp_imp = nonfp_imp[nonfp_imp["importance"] > 0].reset_index(drop=True)

    n_layered = len(MOLECULE_COLUMNS) * FP_SIZE
    n_rdkit_total = len(MOLECULE_COLUMNS) * N_RDKIT
    n_solv = solv_feats.shape[1]

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
            indices.append(n_layered + n_rdkit_total + n_solv + comp_idx * N_QC_FEATURES + qc_idx)

    selected = np.concatenate([np.arange(n_layered), np.array(indices)])
    selected.sort()
    return selected, imp_df


# ═══════════════════════════════════════════
# Optuna
# ═══════════════════════════════════════════
def optuna_objective(trial, X, y):
    params = {
        "objective": "mse",
        "n_estimators": 5000,
        "num_leaves": trial.suggest_categorical("num_leaves", [63, 127, 256]),
        "min_child_samples": trial.suggest_categorical("min_child_samples", [20, 50, 100, 200]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.005, 0.00871, 0.01]),
        "subsample": trial.suggest_categorical("subsample", [0.5, 0.6, 0.7]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.4, 0.5, 0.6]),
        "lambda_l1": trial.suggest_categorical("lambda_l1", [0.0, 0.1, 0.5, 1.0]),
        "lambda_l2": trial.suggest_categorical("lambda_l2", [0.0, 0.1, 0.5, 1.0]),
        "n_jobs": 4,
        "verbosity": -1,
        "importance_type": "gain",
    }

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    r2_list = []
    for train_idx, val_idx in kf.split(X):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=0),
            ],
        )
        preds = model.predict(X[val_idx])
        r2_list.append(float(r2_score(y[val_idx], preds)))
    return np.mean(r2_list)


def train_best_and_save(X, y, rxntype, best_params, output_root):
    config_name = "Layered_RDKit_QC_Solvent"
    rxn_dir = output_root / config_name / f"rxn_{rxntype}"
    rxn_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y), dtype=np.float32)
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        fold_start = perf_counter()
        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=0),
            ],
        )

        preds = model.predict(X[val_idx])
        oof_preds[val_idx] = preds

        r2 = float(r2_score(y[val_idx], preds))
        rmse = float(np.sqrt(mean_squared_error(y[val_idx], preds)))
        mae = float(mean_absolute_error(y[val_idx], preds))

        model.booster_.save_model(str(rxn_dir / f"lgbm_fold{fold}.txt"))

        fold_rows.append({
            "rxntype": rxntype, "fold": fold,
            "r2": r2, "rmse": rmse, "mae": mae,
            "best_iteration": int(model.best_iteration_),
            "seconds": round(perf_counter() - fold_start, 2),
        })
        print(f"    fold {fold}: R2={r2:.6f} RMSE={rmse:.6f} "
              f"iter={model.best_iteration_} ({fold_rows[-1]['seconds']}s)", flush=True)

    np.save(str(rxn_dir / "oof_predictions.npy"), oof_preds)

    r2_vals = [f["r2"] for f in fold_rows]
    print(f"  >>> R2={np.mean(r2_vals):.6f} ± {np.std(r2_vals, ddof=1):.6f}", flush=True)
    return fold_rows


# ═══════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════
def main():
    # 支持命令行指定要跑的 rxntype，如: python optuna-tune-config-c.py 1 2 3
    rxntype_args = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(range(1, 9))

    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent.parent / "data"
    rdkit_dir = dataset_dir / "extra-rdkit"
    solvents_dir = dataset_dir / "solvents"
    drugbank_dir = dataset_dir / "drugbank"
    qc_dirs = [
        dataset_dir / "qm_desc-morfeus-round1",
        dataset_dir / "qm_desc-morfeus",
    ]
    output_root = script_dir.parent / "ckpt-optuna"
    output_root.mkdir(parents=True, exist_ok=True)
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    config_name = "Layered_RDKit_QC_Solvent"

    df = load_train_data(dataset_dir)
    rxn_groups = {int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")}
    print(f"数据总量: {len(df)}, 反应类型数: {len(rxn_groups)}", flush=True)
    print(f"配置 C: Layered FP + Top-{TOP_K} NonFP — Optuna {N_TRIALS} trials", flush=True)
    print(f"指定 rxntypes: {rxntype_args}", flush=True)

    print("\n加载溶剂描述符...", flush=True)
    solvent_lookup = build_solvent_lookup(solvents_dir, drugbank_dir)
    print("\n加载 QC 分子描述符...", flush=True)
    qc_lookups = build_qc_lookup(qc_dirs)

    all_start = perf_counter()
    all_results = []
    c_all_params = {}

    valid_rxntypes = sorted(r for r in rxntype_args if r in rxn_groups)
    for rxntype in valid_rxntypes:
        rxn_df = rxn_groups[rxntype]
        print(f"\n{'='*60}", flush=True)
        print(f"rxntype={rxntype}, n={len(rxn_df)}", flush=True)

        try:
            rdkit_arr = load_rdkit_features(rdkit_dir, rxntype)
        except FileNotFoundError as e:
            print(f"  [跳过] {e}", flush=True)
            continue
        if rdkit_arr.shape[0] != len(rxn_df):
            print(f"  [跳过] 样本数不匹配", flush=True)
            continue

        layered_feats = build_layered_features(rxn_df)
        solv_feats = build_solvent_features(rxn_df, solvent_lookup)
        qc_feats = build_qc_features(rxn_df, qc_lookups)
        X_full = np.concatenate([layered_feats, rdkit_arr, solv_feats, qc_feats], axis=1)
        y = rxn_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        print(f"  全量特征: {X_full.shape[1]} 维", flush=True)

        # 特征选择
        selected_cols, imp_df = select_topk_nonfp(X_full, y, solv_feats, top_k=TOP_K)
        X = X_full[:, selected_cols]
        n_nonfp = len(selected_cols) - FP_SIZE * 5
        print(f"  选中: Layered + Top-{n_nonfp} NonFP = {X.shape[1]} 维", flush=True)

        imp_dir = output_root / config_name / f"rxn_{rxntype}"
        imp_dir.mkdir(parents=True, exist_ok=True)
        imp_df.to_csv(imp_dir / "feature_importance.csv", index=False)

        # Optuna 搜索
        print(f"  开始 Optuna 搜索 ({N_TRIALS} trials)...", flush=True)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        rxn_start = perf_counter()
        study.optimize(lambda trial: optuna_objective(trial, X, y),
                       n_trials=N_TRIALS, show_progress_bar=False)
        optuna_time = perf_counter() - rxn_start

        best_params = study.best_params.copy()
        best_params.update({
            "objective": "mse", "n_estimators": 5000,
            "n_jobs": 4, "verbosity": -1, "importance_type": "gain",
        })

        print(f"  Optuna 最优 CV R2={study.best_value:.6f} ({optuna_time:.1f}s)", flush=True)
        print(f"  最优参数: {study.best_params}", flush=True)

        # 最优参数训练 + 保存
        fold_rows = train_best_and_save(X, y, rxntype, best_params, output_root)

        for f in fold_rows:
            all_results.append({"config": config_name, **f, "optuna_r2": study.best_value})

        c_all_params[str(rxntype)] = {
            "params": best_params,
            "cv_r2": study.best_value,
            "n_features": X.shape[1],
        }

    # 保存结果
    with open(results_dir / f"optuna_{config_name}_best_params.json", "w") as f:
        json.dump(c_all_params, f, indent=2)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_dir / "optuna_config_c_fold_metrics.csv", index=False)

        print(f"\n{'='*60}", flush=True)
        print(f"配置 C 汇总", flush=True)
        print(f"{'='*60}", flush=True)
        for rt in sorted(c_all_params.keys()):
            p = c_all_params[rt]
            print(f"  rxn{rt}: CV R2={p['cv_r2']:.6f} features={p['n_features']}", flush=True)
        avg = np.mean([v["cv_r2"] for v in c_all_params.values()])
        print(f"  平均 CV R2: {avg:.6f}", flush=True)

    print(f"\n模型保存: {output_root}/{config_name}/", flush=True)
    print(f"参数保存: {results_dir}/optuna_{config_name}_best_params.json", flush=True)
    print(f"总耗时: {perf_counter() - all_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
