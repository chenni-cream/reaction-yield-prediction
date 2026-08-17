#!/usr/bin/env python3
"""Generate per-reaction RDKit descriptor matrices without overwriting by default."""
import argparse
from pathlib import Path
import numpy as np
try:
    from .common import (DEFAULT_DATA_DIR, MOLECULE_COLUMNS,
                         calculate_manuscript_rdkit_descriptors,
                         load_rdkit_descriptor_names, load_training_data,
                         protected_outputs)
except ImportError:
    from common import (DEFAULT_DATA_DIR, MOLECULE_COLUMNS,
                        calculate_manuscript_rdkit_descriptors,
                        load_rdkit_descriptor_names, load_training_data,
                        protected_outputs)

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_DATA_DIR/"extra-rdkit")
    parser.add_argument("--overwrite",action="store_true")
    return parser.parse_args()

def main():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    args=parse_args(); data=load_training_data(args.data_dir)
    descriptor_names=load_rdkit_descriptor_names()
    outputs=[]
    for rxn in range(1,9):
        outputs.extend([args.output_dir/f"train-rdkitfeature-rxn{rxn}.gz",
                        args.output_dir/f"train-target-rxn{rxn}.gz"])
    protected_outputs(outputs,args.overwrite)
    for rxn,group in data.groupby("rxntype",sort=True):
        parts=[]
        for column in MOLECULE_COLUMNS:
            rows=[]
            for smiles in group[column].astype(str):
                mol=Chem.MolFromSmiles(smiles)
                if mol is None: raise ValueError(f"Invalid SMILES in {column}: {smiles}")
                rows.append(calculate_manuscript_rdkit_descriptors(mol,descriptor_names))
            parts.append(np.asarray(rows,dtype=float))
        features=np.concatenate(parts,axis=1)
        expected_width=len(MOLECULE_COLUMNS)*len(descriptor_names)
        if features.shape[1] != expected_width:
            raise RuntimeError(f"Expected {expected_width} RDKit features, got {features.shape[1]}")
        np.savetxt(args.output_dir/f"train-rdkitfeature-rxn{int(rxn)}.gz",features,delimiter=",")
        np.savetxt(args.output_dir/f"train-target-rxn{int(rxn)}.gz",group["Yield"].to_numpy(),delimiter=",")
        print(f"rxn_{int(rxn)}: {features.shape}")

if __name__=="__main__": main()
