#!/usr/bin/env python3
"""Generate Morfeus descriptors for Round 2 test molecules without overwriting."""
import argparse, os
from pathlib import Path
import pandas as pd
try:
    from .common import DEFAULT_DATA_DIR, column_name, protected_outputs
    from .morfeus_qmdesc import FEATURE_NAMES, canonicalize, descriptors
except ImportError:
    from common import DEFAULT_DATA_DIR, column_name, protected_outputs
    from morfeus_qmdesc import FEATURE_NAMES, canonicalize, descriptors

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA_DIR); p.add_argument("--output-dir",type=Path,default=DEFAULT_DATA_DIR/"qm_desc-morfeus-round2-test"); p.add_argument("--column","--col",default="0"); p.add_argument("--threads",type=int,default=16); p.add_argument("--overwrite",action="store_true"); return p.parse_args()

def main():
    from rdkit import Chem
    from tqdm import tqdm
    args=parse_args(); column=column_name(str(args.column)); os.environ["OMP_NUM_THREADS"]=str(args.threads)
    output=args.output_dir/f"psikit_{column}.csv.gz"; protected_outputs([output],args.overwrite)
    data=pd.read_csv(args.data_dir/"round2_test_data_with_ans.csv"); lookup={}
    for raw in tqdm(data[column].astype(str),desc=f"Processing {column}"):
        try: components=canonicalize(raw,Chem).split(".")
        except ValueError as exc: print(exc); continue
        for smiles in components:
            if smiles not in lookup:
                try: lookup[smiles]=descriptors(smiles)
                except Exception as exc: print(f"Failed {smiles}: {exc}")
    pd.DataFrame([[s]+v for s,v in lookup.items()],columns=FEATURE_NAMES).to_csv(output,index=False,compression="gzip")
    print(f"Saved {len(lookup)} descriptors to {output}")

if __name__=="__main__": main()
