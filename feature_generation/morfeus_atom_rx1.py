#!/usr/bin/env python3
"""Generate reaction-centre atom descriptors for a selected reaction type."""
import argparse, json, os
from pathlib import Path
import numpy as np
try:
    from .common import DEFAULT_DATA_DIR, protected_outputs
except ImportError:
    from common import DEFAULT_DATA_DIR, protected_outputs

def descriptors(smiles,atom_id):
    import morfeus as mf
    from morfeus import BuriedVolume, Sterimol
    from rdkit import Chem
    from rdkit.Chem import AllChem
    if __package__:
        from .utils.TSEI import calc_TSEI
    else:
        from utils.TSEI import calc_TSEI
    parser=Chem.SmilesParserParams(); parser.removeHs=False
    mol=Chem.MolFromSmiles(smiles,parser)
    if mol is None or AllChem.EmbedMolecule(mol,randomSeed=42)!=0: raise RuntimeError("3D embedding failed")
    AllChem.MMFFOptimizeMolecule(mol)
    lines=Chem.MolToXYZBlock(mol).strip().splitlines()[2:]
    elements=[x.split()[0] for x in lines]; coords=np.asarray([[float(v) for v in x.split()[1:4]] for x in lines])
    atom_ids=[atom_id] if isinstance(atom_id,int) else list(atom_id); one_based=[x+1 for x in atom_ids]
    tsei=[calc_TSEI(mol,[i for i in range(mol.GetNumAtoms()) if i!=j],j) for j in atom_ids]
    buried=[BuriedVolume(elements,coords,i).fraction_buried_volume for i in one_based]
    steric=[]
    if len(one_based)==2:
        value=Sterimol(elements,coords,one_based[0],one_based[1]); steric=[float(value.L_value),float(value.B_1_value),float(value.B_5_value)]
    xtb=mf.XTB(elements,coords); charges=xtb.get_charges(); electrophilic=xtb.get_fukui("electrophilicity"); nucleophilic=xtb.get_fukui("nucleophilicity")
    if len(one_based)==2:
        electronic=[float(charges[one_based[0]]),float(charges[one_based[1]]),float(electrophilic[one_based[0]]),float(electrophilic[one_based[1]]),float(nucleophilic[one_based[0]]),float(nucleophilic[one_based[1]]),float(xtb.get_bond_order(one_based[0],one_based[1]))]
    else: electronic=[float(charges[one_based[0]]),float(electrophilic[one_based[0]]),float(nucleophilic[one_based[0]])]
    return tsei+buried+steric+electronic

def remove_map(mol):
    from rdkit import Chem
    copy=Chem.Mol(mol)
    for atom in copy.GetAtoms(): atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy)

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA_DIR); p.add_argument("--reaction",type=int,default=1); p.add_argument("--threads",type=int,default=1); p.add_argument("--overwrite",action="store_true"); return p.parse_args()

def main():
    from rdkit import Chem
    from tqdm import tqdm
    args=parse_args(); os.environ["OMP_NUM_THREADS"]=str(args.threads); directory=args.data_dir/"reactionmatch"; rxn=args.reaction
    outputs=[directory/f"Reaction_{rxn}_{suffix}atomdata.json" for suffix in ("r1","r2","prod")]; protected_outputs(outputs,args.overwrite)
    reactions=json.loads((directory/f"Reaction_{rxn}.json").read_text()); atom_ids=json.loads((directory/f"Reaction_{rxn}_atomid.json").read_text()); stores=[{}, {}, {}]
    for index,key in tqdm(enumerate(reactions),total=len(reactions)):
        mapped=reactions[key][0]; reactants=mapped.split(">>")[0][1:-1].split("."); product=mapped.split(">>")[1]
        mols=[Chem.MolFromSmarts(reactants[0]),Chem.MolFromSmarts(reactants[1]),Chem.MolFromSmarts(product)]
        labels=("r1","r2","prod")
        for store,mol,label in zip(stores,mols,labels):
            if mol is None: print(f"Invalid SMARTS for {label} row {index}"); continue
            unmapped=remove_map(mol)
            if unmapped not in store:
                try: store[unmapped]=descriptors(Chem.MolToSmiles(mol),atom_ids[key][label])
                except Exception as exc: print(f"Failed {label} row {index}: {exc}")
    for output,store in zip(outputs,stores): output.write_text(json.dumps(store),encoding="utf-8")
    print("Saved: "+", ".join(map(str,outputs)))
if __name__=="__main__": main()
