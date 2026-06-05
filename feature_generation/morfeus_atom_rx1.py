import pandas as pd
from morfeus import read_xyz
from morfeus import XTB
from morfeus import BuriedVolume,Sterimol
import morfeus as mf
import os
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse
import os
import json
from utils.TSEI import calc_TSEI
#定义 morfeus_descriptors 函数计算分子的拓扑立体电子指数（TSEI）、埋藏体积（BV）、立体摩尔参数（Sterimol）和 XTB 相关的电子参数。
irxn=1
save_path='../data/reactionmatch'
with open(os.path.join(save_path,f"Reaction_{irxn}.json"), 'r') as json_file:
    rxn_smiles = json.load(json_file)
with open(os.path.join(save_path,f"Reaction_{irxn}_atomid.json"), 'r') as json_file:
    rx1_atom_dict = json.load(json_file)


ps = Chem.SmilesParserParams()
ps.removeHs = False



def smiles_to_xyz(smile):
    mol=Chem.MolFromSmiles(smile,ps)
    # mol = Chem.AddHs(mol)  # Add hydrogens
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


def morfeus_descriptors(smile,atom_id):
    elements, coordinates, mol = smiles_to_xyz(smile)
    if type(atom_id)==int:
        atom_id2=[atom_id+1]
        atom_id=[atom_id]
    else:
        atom_id2=[atom_id[0]+1,atom_id[1]+1]
    # atom=mol.GetAtomWithIdx(atom_id[0])
    # print(atom.GetSymbol())
    # All start from 1 except TSEI
    # For buried volume    center = coordinates[metal_index - 1]
    TSEI = [calc_TSEI(mol, [i for i in range(len(mol.GetAtoms())) if i != j], j) for j in atom_id]
    bv = [BuriedVolume(elements, coordinates, i).fraction_buried_volume for i in atom_id2]
    steri=[]
    if len(atom_id2)==2:
        # try:
        sterimol = Sterimol(elements, coordinates, atom_id2[0], atom_id2[1])
        steri=[float(sterimol.L_value),float(sterimol.B_1_value),float(sterimol.B_5_value)]
        # except:
        #     print(len(coordinates),len(elements))
        #     xtb = mf.XTB(elements, coordinates)            
        #     print(xtb.get_charges())

    # # Electronic parameters
    xtb = mf.XTB(elements, coordinates)
    if len(atom_id2)==2:        
        xtb_descriptors = [
            float(xtb.get_charges()[atom_id2[0]]),
            float( xtb.get_charges()[atom_id2[1]]),
            float(xtb.get_fukui("electrophilicity")[atom_id2[0]]),
            float(xtb.get_fukui("electrophilicity")[atom_id2[1]]),            
            float(xtb.get_fukui("nucleophilicity")[atom_id2[0]]),
            float(xtb.get_fukui("nucleophilicity")[atom_id2[1]]),            
            float(xtb.get_bond_order(atom_id2[0], atom_id2[1]))
        ]
    else:
        xtb_descriptors = [
            float(xtb.get_charges()[atom_id2[0]]),
            float(xtb.get_fukui("electrophilicity")[atom_id2[0]]),
            float(xtb.get_fukui("nucleophilicity")[atom_id2[0]]),
        ]

    return TSEI+bv+steri+xtb_descriptors

def removemap(mol):    
    # 遍历分子的所有原子，清除 AtomMapNum
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)    
    # 生成没有原子映射的 SMILES
    new_smiles = Chem.MolToSmiles(mol)
    return new_smiles
def smartstosmiles(mol):    
    # 遍历分子的所有原子，清除 AtomMapNum
    # for atom in mol.GetAtoms():
    #     atom.SetAtomMapNum(0)
    
    # 生成没有原子映射的 SMILES
    new_smiles = Chem.MolToSmiles(mol)
    return new_smiles

smi_dict=set()
r1_dict={}
r2_dict={}
prod_dict={}


for i, key in tqdm(enumerate(rxn_smiles), total=len(rxn_smiles)):
    smi, j = rxn_smiles[key]
    
    react_smi=smi.split(">>")[0][1:-1]
    prod_smi=smi.split(">>")[1]    
    react1=react_smi.split(".")[0]
    react2=react_smi.split(".")[1]
    
    r1mol=Chem.MolFromSmarts(react1)
    r2mol=Chem.MolFromSmarts(react2)    
    pmol=Chem.MolFromSmarts(prod_smi)

    r1smiles=smartstosmiles(r1mol)
    r2smiles=smartstosmiles(r2mol)
    psmiles=smartstosmiles(pmol)

    r1nomap=removemap(r1mol)
    r2nomap=removemap(r2mol)
    prodnomap=removemap(pmol)
    
    
    #这里没有做de mapping 
    # break
    if r1nomap not in r1_dict:
        try:
            r1_dict[r1nomap]=morfeus_descriptors(r1smiles,rx1_atom_dict[key]['r1'])
        except:
            print(f"Error Smiles for r1 {i}: {r1nomap}")
    if r2nomap not in r2_dict:
        try:
            r2_dict[r2nomap]=morfeus_descriptors(r2smiles,rx1_atom_dict[key]['r2'])
        except:
            print(f"Error Smiles for r2 {i}: {r2nomap}")
    if prodnomap not in prod_dict:
        try:
            prod_dict[prodnomap]=morfeus_descriptors(psmiles,rx1_atom_dict[key]['prod'])
        except:
            print(f"Error Smiles for prod {i}: {prodnomap}")

with open(os.path.join(save_path,f"Reaction_{irxn}_r1atomdata.json"), 'w') as json_file:
    json.dump(r1_dict, json_file)
with open(os.path.join(save_path,f"Reaction_{irxn}_r2atomdata.json"), 'w') as json_file:
    json.dump(r2_dict, json_file)
with open(os.path.join(save_path,f"Reaction_{irxn}_prodatomdata.json"), 'w') as json_file:
    json.dump(prod_dict, json_file)