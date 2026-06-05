import warnings
warnings.filterwarnings("ignore", message='not removing hydrogen atom with dummy atom neighbors')
import pandas as pd
from psikit import Psikit
from rdkit import Chem
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse



def canonize_line(smis):
    smis=list(set(smis.split('.')))    
    smis=[Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True) for smi in smis]
    return smis
    
columns=['Reactant1','Reactant2','Product','Additive','Solvent']

dataset_dir = '../data'   # Change this to your dataset directory

train_df_round1 = pd.read_csv(f'{dataset_dir}/round1_train_data.csv')
train_df_round2 = pd.read_csv(f'{dataset_dir}/round2_train_data.csv')

train_df_round1['rxntype'] = 1

train_df = pd.concat([train_df_round1, train_df_round2])

grouped = train_df.groupby('rxntype')
train_df_dict = {rxntype: group for rxntype, group in grouped}

print(f'Training set size: {len(train_df)}')


i_rxn=1
icol=3
# all_sets=[]
# for icol in range(len(columns)):
for icol in range(3,4):    
    print(columns[icol])
    colset=set()
    for i_rxn in range(1,9):
    # for i_rxn in range(1,2):
        components = train_df_dict[i_rxn][columns[icol]].to_list()
        num_multicomponents={}
        for i in range(len(components)):
        # for i in range(16):            
            smis=canonize_line(components[i])
            colset.update(smis)
    print(f"Col:{columns[icol]}",len(colset))
    # all_sets.append(colset)

special_additives=[]
for item in colset:
    if len(item) <=6 and "[" in item:
        special_additives.append(item)
recover=['C[Al]','[Li]C', 'C[O-]','CC[O-]','[C-]#N', 'C[Zn]C', '[C]=O', 'C[Ni]C']
for item in recover:
    special_additives.remove(item)
special_smi=set(special_additives)

def canonize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)


# columns=['Reactant1','Reactant2','Product','Additive','Solvent']
col=columns[3]
smiles=train_df_round1[col].tolist()
smi_dict={}
for smi in tqdm(smiles[:], desc="Processing SMILES"):
    smi=canonize(smi)
    smislist=[smi]
    if smi.count('.')>0:
        smislist=smi.split('.')
    for itemsmi in smislist:
        if itemsmi not in smi_dict and itemsmi not in special_smi:
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
df.to_csv(savepath+f'psikit_{col}.csv.gz', index=False, compression='gzip')