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
#定义 morfeus_descriptors 函数，使用 morfeus 库计算分子的色散描述符、溶剂可及表面积描述符和 XTB 相关的电子参数。
os.environ['OMP_NUM_THREADS'] = "16"

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

def smiles_to_xyz(smiles):
    mol = Chem.MolFromSmiles(smiles)
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

    return elements, coordinates


def morfeus_descriptors(smis):
    elements, coordinates = smiles_to_xyz(smis)
    
    # Dispersion
    disp = mf.Dispersion(elements, coordinates)
    disp_descriptors = [disp.area, disp.p_int, disp.volume]
    
    # Solvent accessible surface area
    sasa = mf.SASA(elements, coordinates)
    sasa_descriptors = [sasa.area, sasa.volume]
    
    # Electronic parameters
    xtb = mf.XTB(elements, coordinates)
    xtb_descriptors = [
        # xtb.get_dipole()[0],
        # xtb.get_dipole()[1],
        # xtb.get_dipole()[2],
        float(np.linalg.norm(xtb.get_dipole())),
        xtb.get_ea(corrected=True),
        xtb.get_global_descriptor("electrophilicity", corrected=True),
        xtb.get_global_descriptor("nucleophilicity", corrected=True),
        xtb.get_global_descriptor("electrofugality", corrected=True),
        xtb.get_global_descriptor("nucleofugality", corrected=True),
        float(xtb.get_homo()),
        float(xtb.get_lumo()),
        float(xtb.get_homo()-xtb.get_lumo()),
        xtb.get_ip(corrected=True),
        # xtb.get_charges(),
        # xtb.get_fukui("electrophilicity")
        # xtb.get_fukui("nucleophilicity")
    ]

    return disp_descriptors+sasa_descriptors+xtb_descriptors

columns=['Reactant1','Reactant2','Product','Additive','Solvent']
col=columns[3]
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

data=[]
for line in smi_dict.items():
    data.append([line[0]]+line[1])

namelists=["smile",
           "dipole",
           "electron_affinity",
           "electrophilicity",
           "nucleophilicity",
           "electrofugality",
           "nucleofugality",
           "homo",
           "lumo",
           "homo-lumo",
           "ionization_potential"               
          ]
    
namelists=["smile",
           "disp_area",
           "disp_p_int",
           "disp_volume",
           "sasa_area",
           "sasa_volume",
           "dipole",
           "electron_affinity",
           "electrophilicity",
           "nucleophilicity",
           "electrofugality",
           "nucleofugality",
           "homo",
           "lumo",
           "homo-lumo",
           "ionization_potential"               
          ]

df = pd.DataFrame(data,columns=namelists)
savepath='../data/qm_desc-morfeus-moredescriptors/'
os.makedirs(savepath, exist_ok=True)
df.to_csv(savepath+f'psikit_{col}.csv', index=False)
# df.to_csv(savepath+f'psikit_{col}.csv.gz', index=False, compression='gzip')
