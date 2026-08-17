#!/usr/bin/env python3
"""Generate Morfeus dispersion, SASA, and xTB descriptors for training molecules."""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .common import DEFAULT_DATA_DIR, MOLECULE_COLUMNS, column_name, load_training_data, protected_outputs
except ImportError:
    from common import DEFAULT_DATA_DIR, MOLECULE_COLUMNS, column_name, load_training_data, protected_outputs

FEATURE_NAMES=["smile","disp_area","disp_p_int","disp_volume","sasa_area","sasa_volume","dipole","electron_affinity","electrophilicity","nucleophilicity","electrofugality","nucleofugality","homo","lumo","homo-lumo","ionization_potential"]

def canonicalize(smiles, Chem):
    mol=Chem.MolFromSmiles(str(smiles))
    if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True)

def descriptors(smiles):
    import morfeus as mf
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol,randomSeed=42) != 0: raise RuntimeError(f"3D embedding failed: {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    lines=Chem.MolToXYZBlock(mol).strip().splitlines()[2:]
    elements=[line.split()[0] for line in lines]
    coordinates=np.asarray([[float(x) for x in line.split()[1:4]] for line in lines])
    disp=mf.Dispersion(elements,coordinates); sasa=mf.SASA(elements,coordinates); xtb=mf.XTB(elements,coordinates)
    return [disp.area,disp.p_int,disp.volume,sasa.area,sasa.volume,float(np.linalg.norm(xtb.get_dipole())),xtb.get_ea(corrected=True),xtb.get_global_descriptor("electrophilicity",corrected=True),xtb.get_global_descriptor("nucleophilicity",corrected=True),xtb.get_global_descriptor("electrofugality",corrected=True),xtb.get_global_descriptor("nucleofugality",corrected=True),float(xtb.get_homo()),float(xtb.get_lumo()),float(xtb.get_homo()-xtb.get_lumo()),xtb.get_ip(corrected=True)]

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA_DIR); p.add_argument("--output-dir",type=Path,default=DEFAULT_DATA_DIR/"qm_desc-morfeus-moredescriptors"); p.add_argument("--column",default="Additive"); p.add_argument("--threads",type=int,default=16); p.add_argument("--overwrite",action="store_true"); return p.parse_args()

def main():
    from rdkit import Chem
    from tqdm import tqdm
    args=parse_args(); column=column_name(args.column); os.environ["OMP_NUM_THREADS"]=str(args.threads)
    output=args.output_dir/f"psikit_{column}.csv"; protected_outputs([output],args.overwrite)
    data=load_training_data(args.data_dir); lookup={}
    for raw in tqdm(data[column].astype(str),desc=f"Processing {column}"):
        for smiles in canonicalize(raw,Chem).split("."):
            if smiles not in lookup:
                try: lookup[smiles]=descriptors(smiles)
                except Exception as exc: print(f"Failed {smiles}: {exc}")
    pd.DataFrame([[s]+v for s,v in lookup.items()],columns=FEATURE_NAMES).to_csv(output,index=False)
    print(f"Saved {len(lookup)} descriptors to {output}")

if __name__=="__main__": main()
