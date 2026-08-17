#!/usr/bin/env python3
"""Generate atom-level TSEI and xTB descriptors for one molecular component."""
import argparse, json, os
from pathlib import Path
import numpy as np
try:
    from .common import DEFAULT_DATA_DIR, column_name, load_training_data, protected_outputs
except ImportError:
    from common import DEFAULT_DATA_DIR, column_name, load_training_data, protected_outputs

def descriptors(smiles):
    import morfeus as mf
    from rdkit import Chem
    from rdkit.Chem import AllChem
    if __package__:
        from .utils.TSEI import calc_TSEI
    else:
        from utils.TSEI import calc_TSEI
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol,randomSeed=42)!=0: raise RuntimeError("3D embedding failed")
    AllChem.MMFFOptimizeMolecule(mol)
    lines=Chem.MolToXYZBlock(mol).strip().splitlines()[2:]
    elements=[x.split()[0] for x in lines]; coords=np.asarray([[float(v) for v in x.split()[1:4]] for x in lines])
    n=len(elements); tsei=[calc_TSEI(mol,[i for i in range(n) if i!=j],j) for j in range(n)]
    xtb=mf.XTB(elements,coords)
    xtb_values=[list(xtb.get_charges().values()),list(xtb.get_fukui("electrophilicity").values()),list(xtb.get_fukui("nucleophilicity").values())]
    return {"tsei":tsei,"xtb":np.asarray(xtb_values).tolist()}

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA_DIR); p.add_argument("--output-dir",type=Path,default=DEFAULT_DATA_DIR/"qm_desc-morfeus-atoms-secondary"); p.add_argument("--column",default="Reactant2"); p.add_argument("--threads",type=int,default=1); p.add_argument("--overwrite",action="store_true"); return p.parse_args()

def main():
    from rdkit import Chem
    from tqdm import tqdm
    args=parse_args(); column=column_name(str(args.column)); os.environ["OMP_NUM_THREADS"]=str(args.threads)
    output=args.output_dir/f"psikit_{column}.json"; protected_outputs([output],args.overwrite)
    data=load_training_data(args.data_dir); lookup={}
    for raw in tqdm(data[column].astype(str),desc=f"Processing {column}"):
        mol=Chem.MolFromSmiles(raw)
        if mol is None: print(f"Invalid SMILES: {raw}"); continue
        for smiles in Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True).split("."):
            if smiles not in lookup:
                try: lookup[smiles]=descriptors(smiles)
                except Exception as exc: print(f"Failed {smiles}: {exc}")
    output.write_text(json.dumps(lookup),encoding="utf-8"); print(f"Saved {len(lookup)} descriptors to {output}")
if __name__=="__main__": main()
