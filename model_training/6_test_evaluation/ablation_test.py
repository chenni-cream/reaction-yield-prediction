#!/usr/bin/env python3
"""Compute test set ablation: Full vs w/o each model."""
import gzip, json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import r2_score
try:
    from rdkit.Avalon import pyAvalonTools
except ImportError:
    pyAvalonTools = None
RDLogger.DisableLog("rdApp.*")

MOLECULE_COLUMNS = ["Reactant1","Reactant2","Product","Additive","Solvent"]
TARGET = "Yield"
FP_SIZE = 2048; N_RDKIT = 210; N_FOLDS = 5; TOP_K = 500
SOLV_MAIN_COLS = ["MW (g/mol)","Density (g/mL)","Molar volume (mL/mol)","Refractive index",
    "Mol. refr. pow. (mL/mol)","Dipole moment (D)","Melting point (°C)","Boiling point (°C)",
    "Viscosity (cP)","lnP (partition coeff.)","Vapour pressure (mbar)","Henry's constant","lngamma","neutral"]
MNSOL_COLS = ["alpha","beta","beta**2","eps","gamma","n","phi**2","psi**2"]
DRUGBANK_COLS = ["logP_ALOGPS","logS","logP_ChemAxon","pKa_acid","pKa_base","PSA","Polarizability","H_Acceptor_Count","H_Donor_Count"]
ALL_SOLV_COLS = SOLV_MAIN_COLS + MNSOL_COLS + DRUGBANK_COLS
N_SOLV = len(ALL_SOLV_COLS)
QC_COLS = ["dipole","electron_affinity","electrophilicity","nucleophilicity","electrofugality","nucleofugality","homo","lumo","homo-lumo","ionization_potential"]
N_QC = len(QC_COLS)
RXN = {1:"C–N",2:"Suzuki",3:"Heck",4:"Diels-Alder",5:"SNAr",6:"Sonogashira",7:"Michael",8:"Amidation"}

SD = Path(__file__).resolve().parent
DD = SD.parent.parent / "data"
RD = DD / "extra-rdkit"
MA = SD.parent / "ckpt-searchfp" / "AvalonFingerprint_lgbm"
MB = SD.parent / "ckpt-optuna" / "Avalon_RDKit"
MC = SD.parent / "ckpt-optuna" / "Layered_RDKit_QC_Solvent"
RES = SD.parent / "results"

def bv2np(bv, n):
    a = np.zeros((n,), dtype=np.uint8); DataStructs.ConvertToNumpyArray(bv, a); return a

def build_avalon(df):
    z = np.zeros((FP_SIZE,), dtype=np.uint8); parts = []
    for c in MOLECULE_COLUMNS:
        rows = []
        for s in df[c].fillna("").astype(str):
            m = Chem.MolFromSmiles(s)
            rows.append(z if m is None else bv2np(pyAvalonTools.GetAvalonFP(m, FP_SIZE), FP_SIZE))
        parts.append(np.asarray(rows, dtype=np.float32))
    return np.concatenate(parts, axis=1)

def build_layered(df):
    z = np.zeros((FP_SIZE,), dtype=np.uint8); parts = []
    for c in MOLECULE_COLUMNS:
        rows = []
        for s in df[c].fillna("").astype(str):
            m = Chem.MolFromSmiles(s)
            rows.append(z if m is None else bv2np(Chem.LayeredFingerprint(m, fpSize=FP_SIZE), FP_SIZE))
        parts.append(np.asarray(rows, dtype=np.float32))
    return np.concatenate(parts, axis=1)

def build_rdkit_lookup():
    from rdkit.ML.Descriptors import MoleculeDescriptors
    dn = pd.read_csv(RD / "train-rdkitfeature-Reactant1_feature_names.csv")["FeatureName"].tolist()
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(dn); nd = len(dn)
    jp = {"Reactant1":["test-rdkitfeature-Reactant1.json","train-rdkitfeature-Reactant1.json"],
          "Reactant2":["test-rdkitfeature-Reactant2.json","train-rdkitfeature-Reactant2.json"],
          "Product":["test-rdkitfeature-Product.json","train-rdkitfeature-Product.json"],
          "Additive":["test-rdkitfeature-Additive.json","train-rdkitfeature-Additive-nosplit.json","train-rdkitfeature-Additive.json"],
          "Solvent":["test-rdkitfeature-Solvent.json","train-rdkitfeature-Solvent-nosplit.json","train-rdkitfeature-Solvent.json"]}
    lu = {}
    for col, fns in jp.items():
        lk = {}
        for fn in fns:
            p = RD / fn
            if p.exists():
                with open(p) as f: data = json.load(f)
                for smi, vec in data.items():
                    if smi not in lk:
                        lk[smi] = np.where(np.isfinite(vec), np.array(vec, dtype=np.float32), 0.0)
        lu[col] = (lk, calc, nd)
    return lu

def build_rdkit(df, lu):
    parts = []
    for col in MOLECULE_COLUMNS:
        lk, calc, nd = lu[col]; z = np.zeros((nd,), dtype=np.float32); rows = []
        for s in df[col].fillna("").astype(str):
            if s in lk: rows.append(lk[s])
            else:
                m = Chem.MolFromSmiles(s)
                if m is None: rows.append(z)
                else: v = calc.CalcDescriptors(m); rows.append(np.where(np.isfinite(v), v, 0.0).astype(np.float32))
        parts.append(np.asarray(rows, dtype=np.float32))
    return np.concatenate(parts, axis=1)

def build_solv_lookup():
    m1 = pd.read_csv(DD/"solvents"/"solvent_withsmiles.csv").dropna(subset=["smiles"])
    m2 = pd.read_csv(DD/"solvents"/"MNSol_alldata_withsmiles.csv").dropna(subset=["smiles"])
    m3 = pd.read_csv(DD/"drugbank"/"solvent.csv").dropna(subset=["smiles"])
    mg = m1[["smiles"]+SOLV_MAIN_COLS].copy()
    mg = mg.merge(m2[["smiles"]+MNSOL_COLS], on="smiles", how="outer")
    mg = mg.merge(m3[["smiles"]+DRUGBANK_COLS], on="smiles", how="outer")
    lu = {}
    for _, r in mg.iterrows():
        smi = r["smiles"]
        if pd.isna(smi): continue
        lu[str(smi)] = r[ALL_SOLV_COLS].to_numpy(dtype=np.float64)
    aa = np.array(list(lu.values())); cm = np.nanmean(aa, axis=0); cm[np.isnan(cm)] = 0.0
    for smi in lu:
        v = lu[smi]; nm = np.isnan(v)
        if nm.any():
            v[nm] = cm[nm]; lu[smi] = v
    return lu

def build_solv(df, sl):
    z = np.zeros(N_SOLV, dtype=np.float32); cf = []
    for s in df["Solvent"].fillna("").astype(str):
        vs = [sl[p.strip()].astype(np.float32) for p in s.split(".") if p.strip() in sl]
        cf.append(np.mean(vs, axis=0) if vs else z)
    return np.asarray(cf, dtype=np.float32)

def build_qc_lookup():
    fm = {"Reactant1":"psikit_Reactant1.csv.gz","Reactant2":"psikit_Reactant2.csv.gz",
          "Product":"psikit_Product.csv.gz","Additive":"psikit_Additive.csv.gz","Solvent":"psikit_Solvent.csv.gz"}
    qd = [DD/d for d in ["qm_desc-morfeus-round1","qm_desc-morfeus","qm_desc-morfeus-round1-test","qm_desc-morfeus-round2-test"]]
    lu = {}
    for col, fn in fm.items():
        lk = {}
        for d in qd:
            p = d / fn
            if p.exists():
                with gzip.open(p, "rt") as f: df_qc = pd.read_csv(f)
                for _, r in df_qc.iterrows():
                    smi = r["smile"]
                    if smi not in lk: lk[smi] = r[QC_COLS].to_numpy(dtype=np.float32)
        lu[col] = lk
    return lu

def build_qc(df, ql):
    z = np.zeros(N_QC, dtype=np.float32); parts = []
    for col in MOLECULE_COLUMNS:
        parts.append(np.asarray([ql[col].get(s, z) for s in df[col].fillna("").astype(str)], dtype=np.float32))
    return np.concatenate(parts, axis=1)

def sel_B(rxntype):
    imp = pd.read_csv(MB / f"rxn_{rxntype}" / "feature_importance.csv")
    ri = imp[imp["feature"].str.startswith("RDKit_")]; ri = ri[ri["importance"] > 0]
    na = len(MOLECULE_COLUMNS) * FP_SIZE; idx = []
    for f in ri["feature"].iloc[:TOP_K]:
        p = f.split("_", 2); idx.append(na + MOLECULE_COLUMNS.index(p[1]) * N_RDKIT + int(p[2]))
    s = np.concatenate([np.arange(na), np.array(idx)]); s.sort(); return s

def sel_C(rxntype):
    imp = pd.read_csv(MC / f"rxn_{rxntype}" / "feature_importance.csv")
    nf = imp[~imp["feature"].str.startswith("Layered_")]
    nf = nf[nf["importance"] > 0].reset_index(drop=True)
    nl = len(MOLECULE_COLUMNS) * FP_SIZE; nrt = len(MOLECULE_COLUMNS) * N_RDKIT; idx = []
    for f in nf["feature"].iloc[:TOP_K]:
        if f.startswith("RDKit_"):
            p = f.split("_", 2); idx.append(nl + MOLECULE_COLUMNS.index(p[1]) * N_RDKIT + int(p[2]))
        elif f.startswith("Solvent_"):
            idx.append(nl + nrt + ALL_SOLV_COLS.index(f[len("Solvent_"):]))
        else:
            p = f.split("_", 2); idx.append(nl + nrt + N_SOLV + MOLECULE_COLUMNS.index(p[1]) * N_QC + QC_COLS.index(p[2]))
    s = np.concatenate([np.arange(nl), np.array(idx)]); s.sort(); return s

def predict(X, model_dir):
    ps = []
    for i in range(1, N_FOLDS + 1):
        p = model_dir / f"lgbm_fold{i}.txt"
        if p.exists(): ps.append(lgb.Booster(model_file=str(p)).predict(X))
    return np.mean(ps, axis=0)

def main():
    r1 = pd.read_csv(DD / "round1_test_data_with_ans.csv"); r1["rxntype"] = 1
    r2 = pd.read_csv(DD / "round2_test_data_with_ans.csv"); r2["rxntype"] = pd.to_numeric(r2["rxntype"], errors="coerce").astype(int)
    df = pd.concat([r1, r2], ignore_index=True)
    rxn = {int(k): v.reset_index(drop=True) for k, v in df.groupby("rxntype")}

    print("Loading lookups...")
    rdkit_lu = build_rdkit_lookup()
    solv_lu = build_solv_lookup()
    qc_lu = build_qc_lookup()

    rows = []
    all_y, all_pA, all_pB, all_pC = [], [], [], []

    for rt in range(1, 9):
        rdf = rxn[rt]; y = rdf[TARGET].values.astype(np.float32)
        rn = RXN[rt]
        print(f"  rxn{rt} ({rn}), n={len(y)}")

        avalon = build_avalon(rdf)
        rdkit = build_rdkit(rdf, rdkit_lu)
        layered = build_layered(rdf)
        solv = build_solv(rdf, solv_lu)
        qc = build_qc(rdf, qc_lu)

        pA = predict(avalon, MA / f"rxn_{rt}")
        sB = sel_B(rt); pB = predict(np.concatenate([avalon, rdkit], axis=1)[:, sB], MB / f"rxn_{rt}")
        sC = sel_C(rt); pC = predict(np.concatenate([layered, rdkit, solv, qc], axis=1)[:, sC], MC / f"rxn_{rt}")

        all_y.append(y); all_pA.append(pA); all_pB.append(pB); all_pC.append(pC)

        rows.append({
            'rxntype': rt, 'rxn_name': rn, 'n': len(y),
            'Test_R2_Avalon': r2_score(y, pA),
            'Test_R2_Avalon_RDKit': r2_score(y, pB),
            'Test_R2_Layered_NonFP': r2_score(y, pC),
            'Test_R2_Full': r2_score(y, (pA+pB+pC)/3),
            'Test_R2_wo_Avalon': r2_score(y, (pB+pC)/2),
            'Test_R2_wo_Avalon_RDKit': r2_score(y, (pA+pC)/2),
            'Test_R2_wo_Layered_NonFP': r2_score(y, (pA+pB)/2),
        })

    ay = np.concatenate(all_y); aA = np.concatenate(all_pA); aB = np.concatenate(all_pB); aC = np.concatenate(all_pC)
    rows.append({
        'rxntype': 'Overall', 'rxn_name': '', 'n': len(ay),
        'Test_R2_Avalon': r2_score(ay, aA),
        'Test_R2_Avalon_RDKit': r2_score(ay, aB),
        'Test_R2_Layered_NonFP': r2_score(ay, aC),
        'Test_R2_Full': r2_score(ay, (aA+aB+aC)/3),
        'Test_R2_wo_Avalon': r2_score(ay, (aB+aC)/2),
        'Test_R2_wo_Avalon_RDKit': r2_score(ay, (aA+aC)/2),
        'Test_R2_wo_Layered_NonFP': r2_score(ay, (aA+aB)/2),
    })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(RES / "test_ablation.csv", index=False)

    print(f"\n{'='*110}")
    print('Test Set Ablation (Simple Average)')
    print(f"{'Reaction':<14} {'Full':>10} {'w/o Avalon':>12} {'w/o Avalon+RDKit':>18} {'w/o Layered+NonFP':>20}")
    print('-'*110)
    for _, r in df_res.iterrows():
        nm = str(r['rxn_name']) if r['rxn_name'] else 'Overall'
        print(f"{nm:<14} {r['Test_R2_Full']:>10.4f} {r['Test_R2_wo_Avalon']:>12.4f} {r['Test_R2_wo_Avalon_RDKit']:>18.4f} {r['Test_R2_wo_Layered_NonFP']:>20.4f}")

    print(f"\n{'='*110}")
    print('Individual Model Test R2')
    print(f"{'Reaction':<14} {'Avalon':>10} {'Avalon+RDKit':>14} {'Layered+NonFP':>16}")
    print('-'*110)
    for _, r in df_res.iterrows():
        nm = str(r['rxn_name']) if r['rxn_name'] else 'Overall'
        print(f"{nm:<14} {r['Test_R2_Avalon']:>10.4f} {r['Test_R2_Avalon_RDKit']:>14.4f} {r['Test_R2_Layered_NonFP']:>16.4f}")

    print(f"\nSaved: {RES / 'test_ablation.csv'}")

if __name__ == "__main__":
    main()
