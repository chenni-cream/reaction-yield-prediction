#!/usr/bin/env python3
"""Generate Psi4/Psikit descriptors with repository-relative paths and safe outputs."""
import argparse
from pathlib import Path
import pandas as pd
try:
    from .common import DEFAULT_DATA_DIR, column_name, load_training_data, protected_outputs
except ImportError:
    from common import DEFAULT_DATA_DIR, column_name, load_training_data, protected_outputs

RECOVER_ADDITIVES={"C[Al]","[Li]C","C[O-]","CC[O-]","[C-]#N","C[Zn]C","[C]=O","C[Ni]C"}

def calculate(smiles):
    from psikit import Psikit
    pk=Psikit(); pk.read_from_smiles(smiles)
    return {"scf_energy":pk.energy(basis_sets="scf/3-21g"),"homo":pk.HOMO,"lumo":pk.LUMO,"dipole_all":pk.dipolemoment[3]}

def generate(data, column, output, overwrite=False, skip_smiles=None):
    from rdkit import Chem
    from tqdm import tqdm
    protected_outputs([output],overwrite); rows={}; skip_smiles=skip_smiles or set()
    source=data[column].astype(str)
    for raw in tqdm(source,desc=f"Processing {column}"):
        mol=Chem.MolFromSmiles(raw)
        if mol is None: print(f"Invalid SMILES: {raw}"); continue
        canonical=Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True)
        for smiles in canonical.split("."):
            if smiles in rows or smiles in skip_smiles: continue
            try: rows[smiles]=calculate(smiles)
            except Exception as exc: print(f"Failed {smiles}: {exc}")
    records=[{"smi":s,**values} for s,values in rows.items()]
    pd.DataFrame(records,columns=["smi","scf_energy","homo","lumo","dipole_all"]).to_csv(output,index=False,compression="gzip")
    print(f"Saved {len(records)} descriptors to {output}")

def parse_args(default_column="Reactant2"):
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA_DIR); p.add_argument("--output-dir",type=Path,default=DEFAULT_DATA_DIR/"qm_desc"); p.add_argument("--column",default=default_column); p.add_argument("--round1-only",action="store_true"); p.add_argument("--skip-special-additives",action="store_true"); p.add_argument("--overwrite",action="store_true"); return p.parse_args()

def run(default_column="Reactant2",force_round1=False,skip_special=False):
    args=parse_args(default_column); column=column_name(str(args.column)); data=load_training_data(args.data_dir)
    if args.round1_only or force_round1: data=data[data["rxntype"]==1]
    output=args.output_dir/f"psikit_{column}.csv.gz"
    excluded=set()
    if args.skip_special_additives or skip_special:
        from rdkit import Chem
        for raw in data["Additive"].astype(str):
            mol=Chem.MolFromSmiles(raw)
            if mol is None: continue
            canonical=Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True)
            excluded.update(smiles for smiles in canonical.split(".") if len(smiles)<=6 and "[" in smiles)
        excluded.difference_update(RECOVER_ADDITIVES)
        print(f"Skipping {len(excluded)} dynamically identified special additives")
    generate(data,column,output,args.overwrite,excluded)

def main(): run()
if __name__=="__main__": main()
