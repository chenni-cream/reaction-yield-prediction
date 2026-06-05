import pandas as pd
from morfeus import XTB
import morfeus as mf
import os
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
import numpy as np

os.environ['OMP_NUM_THREADS'] = "16"

dataset_dir = '../data'

# Read round2 test data
test_df = pd.read_csv(f'{dataset_dir}/round2_test_data_with_ans.csv')
print(f'Round2 test set size: {len(test_df)}')


def canonize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def smiles_to_xyz(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    xyz_string = Chem.MolToXYZBlock(mol)

    data_list = xyz_string.strip().split("\n")[2:]
    elements = []
    coordinates = []

    for line in data_list:
        parts = line.split()
        elements.append(parts[0])
        coordinates.append([float(parts[i]) for i in range(1, 4)])

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
        float(np.linalg.norm(xtb.get_dipole())),
        xtb.get_ea(corrected=True),
        xtb.get_global_descriptor("electrophilicity", corrected=True),
        xtb.get_global_descriptor("nucleophilicity", corrected=True),
        xtb.get_global_descriptor("electrofugality", corrected=True),
        xtb.get_global_descriptor("nucleofugality", corrected=True),
        float(xtb.get_homo()),
        float(xtb.get_lumo()),
        float(xtb.get_homo() - xtb.get_lumo()),
        xtb.get_ip(corrected=True),
    ]

    return disp_descriptors + sasa_descriptors + xtb_descriptors


namelists = [
    "smile",
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
    "ionization_potential",
]

columns = ['Reactant1', 'Reactant2', 'Product', 'Additive', 'Solvent']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--col', type=int, default=0, help='Column index 0-4')
args = parser.parse_args()

col = columns[args.col]
print(f"Treating {col}")
smiles = test_df[col].tolist()
smi_dict = {}
for smi in tqdm(smiles[:], desc="Processing SMILES"):
    canon = canonize(smi)
    if canon is None:
        print(f"Cannot parse SMILES: {smi}")
        continue
    smislist = [canon]
    if canon.count('.') > 0:
        smislist = canon.split('.')
    for itemsmi in smislist:
        if itemsmi not in smi_dict:
            try:
                smi_dict[itemsmi] = morfeus_descriptors(itemsmi)
            except Exception as e:
                print(f"Error Smiles: {itemsmi}, Error: {e}")

data = []
for line in smi_dict.items():
    data.append([line[0]] + line[1])

df = pd.DataFrame(data, columns=namelists)
savepath = f'../data/qm_desc-morfeus-round2-test/'
os.makedirs(savepath, exist_ok=True)
df.to_csv(savepath + f'psikit_{col}.csv.gz', index=False, compression='gzip')
print(f"Saved {len(df)} descriptors to {savepath}psikit_{col}.csv.gz")
