#!/usr/bin/env python3
"""Generate per-component RDKit descriptor lookup JSON files safely."""
import argparse, json
from pathlib import Path
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
    parser.add_argument("--column",choices=MOLECULE_COLUMNS,action="append",dest="columns")
    parser.add_argument("--overwrite",action="store_true")
    return parser.parse_args()

def main():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    args=parse_args(); columns=args.columns or MOLECULE_COLUMNS
    descriptor_names=load_rdkit_descriptor_names()
    outputs=[args.output_dir/f"train-rdkitfeature-{column}.json" for column in columns]
    protected_outputs(outputs,args.overwrite)
    data=load_training_data(args.data_dir)
    for column,output in zip(columns,outputs):
        lookup={}
        for raw in data[column].astype(str):
            mol=Chem.MolFromSmiles(raw)
            if mol is None: raise ValueError(f"Invalid SMILES in {column}: {raw}")
            canonical=Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True)
            for smiles in canonical.split("."):
                if smiles not in lookup:
                    component=Chem.MolFromSmiles(smiles)
                    lookup[smiles]=calculate_manuscript_rdkit_descriptors(component,descriptor_names)
        output.write_text(json.dumps(lookup),encoding="utf-8")
        print(f"{column}: {len(lookup)} unique components -> {output}")

if __name__=="__main__": main()
