"""
Extract from the CSV file of the importance of features for 8 types of reactions:
1. Fingerprint-like features: The key fragments ranked from the top 50 to 100 for each reaction type
2. QC/physical features: General features that appear in ≥ 3 types of reactions (even if ranked up to 150, they are retained)
3. Solvent physical features: General features that appear in ≥ 3 types of reactions 
"""

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict

# ============================================================
# 路径配置
# ============================================================
BASE = str(Path(__file__).resolve().parent.parent / "ckpt-ablation-solvent-qc")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Helper functions
# ============================================================
def classify_feature(name):
    """将特征名分为 fingerprint / QC / Solvent / other"""
    if name.startswith("QC_"):
        return "QC"
    elif name.startswith("Solvent_") or name in ("Solvent_H_Donor", "Solvent_H_Acceptor"):
        return "Solvent"
    elif name.startswith(("RDKit_", "Layered_", "Morgan_")):
        return "fingerprint"
    else:
        return "other"


def fingerprint_subtype(name):
    """指纹子类型: RDKit / Layered / Morgan"""
    if name.startswith("RDKit_"):
        return "RDKit"
    elif name.startswith("Layered_"):
        return "Layered"
    elif name.startswith("Morgan_"):
        return "Morgan"
    return "unknown"


def qc_base_property(name):
    """QC特征 -> 物理属性名, e.g. QC_Product_homo -> homo"""
    parts = name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return name


def qc_molecule(name):
    """QC特征 -> 分子类型, e.g. QC_Product_homo -> Product"""
    parts = name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


# ============================================================
# 1. 加载全部8个反应的CSV，添加rank和category
# ============================================================
dfs = {}
for i in range(1, 9):
    path = os.path.join(BASE, f"rxn_{i}", "feature_importance_full.csv")
    df = pd.read_csv(path)
    df.columns = ["feature", "importance"]
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["rxn"] = f"rxn_{i}"
    df["category"] = df["feature"].apply(classify_feature)
    dfs[f"rxn_{i}"] = df


# ============================================================
# 2. 指纹类特征：各反应排名前100
# ============================================================
print("=" * 80)
print("1. 指纹类特征 (Fingerprint): 各反应类型中排名前100的关键片段")
print("=" * 80)

fp_all_top100 = defaultdict(set)  # feature -> set of rxns where it's in top 100

for rxn, df in dfs.items():
    df_fp = df[df["category"] == "fingerprint"]
    top100 = df_fp[df_fp["rank"] <= 100]
    for feat in top100["feature"].tolist():
        fp_all_top100[feat].add(rxn)

for rxn in [f"rxn_{i}" for i in range(1, 9)]:
    df_fp = dfs[rxn][dfs[rxn]["category"] == "fingerprint"]
    top100 = df_fp[df_fp["rank"] <= 100].copy()
    top100["fp_type"] = top100["feature"].apply(fingerprint_subtype)
    print(f"\n{rxn}: 共 {len(top100)} 个指纹特征在前100名")
    for fp_type in ["RDKit", "Layered"]:
        subset = top100[top100["fp_type"] == fp_type]
        print(f"  {fp_type}: {len(subset)} 个")
        for _, row in subset.head(10).iterrows():
            print(f"    {row['feature']}  (rank={row['rank']}, importance={row['importance']:.2f})")
        if len(subset) > 10:
            print(f"    ... 及其他 {len(subset)-10} 个")

# ============================================================
# 跨反应通用指纹特征 (≥3类反应排名前100)
# ============================================================
print("\n\n--- 跨反应通用指纹特征 (在≥3类反应中排名前100) ---")
cross_fp = {feat: rxns for feat, rxns in fp_all_top100.items() if len(rxns) >= 3}
cross_fp_sorted = sorted(cross_fp.items(), key=lambda x: (-len(x[1]), x[0]))

if cross_fp_sorted:
    print(f"\n共 {len(cross_fp_sorted)} 个指纹片段在≥3类反应中均排名前100:")
    for feat, rxns in cross_fp_sorted[:60]:
        rxns_str = ", ".join(sorted(rxns))
        ranks = [dfs[r][dfs[r]["feature"] == feat]["rank"].values[0] for r in rxns]
        best_rank = min(ranks)
        avg_rank = sum(ranks) / len(ranks)
        print(f"  {feat}: {len(rxns)}类反应 ({rxns_str}), 最佳排名={best_rank}, 平均排名={avg_rank:.0f}")
    if len(cross_fp_sorted) > 60:
        print(f"  ... 及其他 {len(cross_fp_sorted) - 60} 个特征")
else:
    print("  无指纹特征在≥3类反应中均排名前100")


# ============================================================
# 3. QC/物理特征：跨反应通用特征 (≥3类反应中出现即保留)
# ============================================================
print("\n\n" + "=" * 80)
print("2. 物理/QC类特征: 跨反应通用特征 (≥3类反应中出现即保留)")
print("=" * 80)

qc_cross = defaultdict(list)  # (property, molecule) -> [(rxn, rank, imp)]
for rxn, df in dfs.items():
    df_qc = df[df["category"] == "QC"]
    for _, row in df_qc.iterrows():
        prop = qc_base_property(row["feature"])
        mol = qc_molecule(row["feature"])
        qc_cross[(prop, mol)].append((rxn, row["rank"], row["importance"]))

# 按物理属性聚合（不区分分子）
qc_prop_cross = defaultdict(list)  # property -> [(rxn, rank, imp, mol)]
for (prop, mol), entries in qc_cross.items():
    for rxn, rank, imp in entries:
        qc_prop_cross[prop].append((rxn, rank, imp, mol))

print(f"\n{'属性':<25} {'出现反应数':<10} {'排名范围':<20} {'是否保留(≥3类)':<15}")
print("-" * 80)

qc_keep_features = []
for prop in sorted(qc_prop_cross.keys()):
    entries = qc_prop_cross[prop]
    rxns_present = set(e[0] for e in entries)
    ranks = [e[1] for e in entries]
    keep = len(rxns_present) >= 3
    rank_range = f"{min(ranks)}-{max(ranks)}"
    keep_str = "✓ 保留" if keep else "✗ 不保留"
    print(f"{prop:<25} {len(rxns_present)}/8反应    排名{rank_range:<12} {keep_str}")
    if keep:
        for rxn, rank, imp, mol in entries:
            qc_keep_features.append((f"QC_{mol}_{prop}", rxn, rank, imp))

# QC保留特征详情
print("\n\n--- 保留的QC特征详情 ---")
for prop in sorted(qc_prop_cross.keys()):
    entries = qc_prop_cross[prop]
    rxns_present = set(e[0] for e in entries)
    if len(rxns_present) < 3:
        continue
    print(f"\n属性: {prop} (出现在 {len(rxns_present)} 类反应)")
    for (p, mol), mol_entries in sorted(qc_cross.items()):
        if p != prop:
            continue
        print(f"  分子类型: {mol}")
        for rxn, rank, imp in sorted(mol_entries, key=lambda x: x[0]):
            print(f"    {rxn}: 排名={rank}, importance={imp:.2f}")


# ============================================================
# 4. 溶剂物理特征
# ============================================================
print("\n\n--- 溶剂物理特征 (Solvent_开头) ---")
solv_cross = defaultdict(list)
for rxn, df in dfs.items():
    df_solv = df[df["category"] == "Solvent"]
    for _, row in df_solv.iterrows():
        solv_cross[row["feature"]].append((rxn, row["rank"], row["importance"]))

for feat, entries in sorted(solv_cross.items()):
    rxns = set(e[0] for e in entries)
    ranks = [e[1] for e in entries]
    keep = len(rxns) >= 3
    print(f"  {feat}: {len(rxns)}/8反应, 排名{min(ranks)}-{max(ranks)}, {'✓保留' if keep else '✗不保留'}")


# ============================================================
# 5. 汇总
# ============================================================
print("\n\n" + "=" * 80)
print("3. 最终特征选择汇总")
print("=" * 80)

# 指纹特征去重
all_fp_keep = set()
for rxn in [f"rxn_{i}" for i in range(1, 9)]:
    df_fp = dfs[rxn][dfs[rxn]["category"] == "fingerprint"]
    top100 = df_fp[df_fp["rank"] <= 100]
    all_fp_keep.update(top100["feature"].tolist())

# QC特征
all_qc_keep = set()
for prop in qc_prop_cross:
    entries = qc_prop_cross[prop]
    rxns = set(e[0] for e in entries)
    if len(rxns) >= 3:
        for rxn, rank, imp, mol in entries:
            all_qc_keep.add(f"QC_{mol}_{prop}")

# 溶剂特征
all_solv_keep = set()
for feat, entries in solv_cross.items():
    if len(set(e[0] for e in entries)) >= 3:
        all_solv_keep.add(feat)

print(f"\n指纹特征 (各反应前100, 去重): {len(all_fp_keep)} 个")
print(f"  其中RDKit: {sum(1 for f in all_fp_keep if f.startswith('RDKit_'))} 个")
print(f"  其中Layered: {sum(1 for f in all_fp_keep if f.startswith('Layered_'))} 个")
print(f"QC特征 (≥3类反应): {len(all_qc_keep)} 个")
print(f"溶剂物理特征 (≥3类反应): {len(all_solv_keep)} 个")
print(f"总计: {len(all_fp_keep) + len(all_qc_keep) + len(all_solv_keep)} 个特征")


# ============================================================
# 6. 保存结果
# ============================================================
# --- 指纹特征 ---
fp_rows = []
for feat in sorted(all_fp_keep):
    appearances = []
    for rxn in [f"rxn_{i}" for i in range(1, 9)]:
        match = dfs[rxn][dfs[rxn]["feature"] == feat]
        if len(match) > 0 and match.iloc[0]["rank"] <= 100:
            appearances.append(f"{rxn}(rank={int(match.iloc[0]['rank'])})")
    fp_rows.append({
        "feature": feat,
        "fp_type": fingerprint_subtype(feat),
        "n_reactions_in_top100": len(fp_all_top100[feat]),
        "reactions": "; ".join(appearances)
    })
pd.DataFrame(fp_rows).to_csv(os.path.join(OUTPUT_DIR, "selected_fingerprint_features.csv"), index=False)

# --- QC特征 ---
qc_rows = []
for feat in sorted(all_qc_keep):
    parts = feat.split("_")
    mol = parts[1]
    prop = "_".join(parts[2:])
    rxns_with_rank = []
    for rxn in [f"rxn_{i}" for i in range(1, 9)]:
        match = dfs[rxn][dfs[rxn]["feature"] == feat]
        if len(match) > 0:
            rxns_with_rank.append(f"{rxn}(rank={int(match.iloc[0]['rank'])})")
    qc_rows.append({
        "feature": feat,
        "property": prop,
        "molecule": mol,
        "n_reactions": len(rxns_with_rank),
        "reactions": "; ".join(rxns_with_rank)
    })
pd.DataFrame(qc_rows).to_csv(os.path.join(OUTPUT_DIR, "selected_qc_features.csv"), index=False)

# --- 合并特征列表 ---
all_features = sorted(all_fp_keep) + sorted(all_qc_keep) + sorted(all_solv_keep)
pd.DataFrame({
    "feature": all_features,
    "category": [classify_feature(f) for f in all_features]
}).to_csv(os.path.join(OUTPUT_DIR, "selected_features_combined.csv"), index=False)

print(f"\n结果已保存到 {OUTPUT_DIR}/:")
print(f"  selected_fingerprint_features.csv  ({len(all_fp_keep)} 个指纹特征)")
print(f"  selected_qc_features.csv           ({len(all_qc_keep)} 个QC特征)")
print(f"  selected_features_combined.csv      ({len(all_features)} 个全部特征)")
