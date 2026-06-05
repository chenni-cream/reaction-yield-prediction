import pandas as pd
from psikit import Psikit
from rdkit import Chem
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os


dataset_dir = '../data'   # Change this to your dataset directory

train_df_round1 = pd.read_csv(f'{dataset_dir}/round1_train_data.csv')
train_df_round2 = pd.read_csv(f'{dataset_dir}/round2_train_data.csv')

train_df_round1['rxntype'] = 1

train_df = pd.concat([train_df_round1, train_df_round2])

print(f'Training set size: {len(train_df)}')


def canonize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)

# special_smi={'[Ar]','[OH-]','[Li+]'}
columns=['Reactant1','Reactant2','Product','Additive','Solvent']
col=columns[4]
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
                pk = Psikit()
                pk.read_from_smiles(itemsmi)
                # pk.optimize()
                smi_dict[itemsmi]={
                            "scf_enery":pk.energy(basis_sets="scf/3-21g"),
                            #    "scf_enery":pk.energy(basis_sets="scf/sto-3g"),
                            "homo":pk.HOMO,
                            "lumo":pk.LUMO,
                            "dipole_all":pk.dipolemoment[3]                      
                            }
            except:
                print(f"Error Smiles: {itemsmi}")


datalist=[]
for line in smi_dict.items():
    datalist.append({"smi":line[0],"scf_energy":line[1]['scf_enery'],
                     "homo":line[1]['homo'],
                     "lumo":line[1]['lumo'],
                     "dipole_all":line[1]['dipole_all']        
    })
df = pd.DataFrame(datalist)
savepath='../data/qm_desc/'
os.makedirs(savepath, exist_ok=True)
df.to_csv(savepath+f'psikit_{col}.csv.gz', index=False, compression='gzip')