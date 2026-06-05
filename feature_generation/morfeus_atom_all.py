import pandas as pd
from morfeus import read_xyz
from morfeus import XTB
import morfeus as mf
import os
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse
import json
from Utils.TSEI import calc_TSEI
from morfeus import BuriedVolume,Sterimol

# os.environ['OMP_NUM_THREADS'] = "16"

dataset_dir = '../data'   # Change this to your dataset directory

train_df_round1 = pd.read_csv(f'{dataset_dir}/round1_train_data.csv')
train_df_round2 = pd.read_csv(f'{dataset_dir}/round2_train_data.csv')

train_df_round1['rxntype'] = 1

train_df = pd.concat([train_df_round1, train_df_round2])

test_df = pd.read_csv(f'{dataset_dir}/round1_test_data.csv')
test_df['rxntype'] = 1

print(f'Training set size: {len(train_df)}')


def canonize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)

def smiles_to_xyz(smile):
    mol=Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)  # Add hydrogens
    AllChem.EmbedMolecule(mol)  # Generate 3D coordinates
    AllChem.MMFFOptimizeMolecule(mol)  # Optimize geometry
    xyz_string = Chem.MolToXYZBlock(mol)  # Convert to XYZ format
    # Split XYZ string into lines, remove first 2
    data_list = xyz_string.strip().split("\n")[2:]

    # Extract elements and coordinates from each line
    elements = []
    coordinates = []

    for line in data_list:
        parts = line.split()
        elements.append(parts[0])  # Atom label
        coordinates.append(
            [float(parts[i]) for i in range(1, 4)]
        )  # X, Y, Z coordinates

    # Convert coordinates to NumPy array
    coordinates = np.array(coordinates)

    return elements, coordinates, mol

#计算分子描述符：定义 morfeus_descriptors 函数，使用 morfeus 库计算分子的拓扑立体电子指数（TSEI）和 XTB 相关的电子参数。
def morfeus_descriptors(smile):
    elements, coordinates, mol = smiles_to_xyz(smile)
    # print(len(elements),mol.GetNumAtoms())
    # atom=mol.GetAtomWithIdx(atom_id[0])
    # print(atom.GetSymbol())
    # All start from 1 except TSEI
    # For buried volume    center = coordinates[metal_index - 1]
    res_dict={}
    atoms_num = len(elements)
    TSEI =  [calc_TSEI(mol, [i for i in range(atoms_num) if i != j], j) for j in range(atoms_num)]
    # bv = [BuriedVolume(elements, coordinates, i).fraction_buried_volume for i in range(1,atoms_num+1)]  #太慢了 Solvent或许可以加
    res_dict['tsei']=TSEI
    # res_dict['bv']=bv

    # # Electronic parameters
    xtb = mf.XTB(elements, coordinates)
    xtb_res = [
        list(xtb.get_charges().values()),
        list(xtb.get_fukui("electrophilicity").values()),
        list(xtb.get_fukui("nucleophilicity").values())
    ]

    res_dict['xtb'] = np.array(xtb_res).tolist()
    return res_dict

columns=['Reactant1','Reactant2','Product','Additive','Solvent']
col=columns[1]
print(f"Treating {col}")
smiles=train_df[col].tolist()
smi_dict={}
for smi in tqdm(smiles[:], desc="Processing SMILES"):
    smi=canonize(smi)
    smislist=[smi]
    if smi.count('.')>0:
        smislist=smi.split('.')
    for itemsmi in smislist:
        if itemsmi not in smi_dict:
            try:
                smi_dict[itemsmi]=morfeus_descriptors(itemsmi)
            except:
                print(f"Error Smiles: {itemsmi}")

savepath='../data/qm_desc-morfeus-atoms-secondary/'
os.makedirs(savepath, exist_ok=True)
with open(os.path.join(savepath,f"psikit_{col}.json"), 'w') as json_file:
    json.dump(smi_dict, json_file)