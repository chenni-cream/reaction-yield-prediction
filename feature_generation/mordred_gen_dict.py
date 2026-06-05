from skfp.fingerprints import MACCSFingerprint, MordredFingerprint, EStateFingerprint, AutocorrFingerprint, GhoseCrippenFingerprint, MQNsFingerprint, LaggnerFingerprint
from rdkit.Chem import Descriptors
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger,Chem
import numpy as np
import os
import json

RDLogger.DisableLog('rdApp.*')

class featurizers_rdkit():
    def transform(self,smis):
        features=[]
        for smi in smis:
            vals = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smi))
            features.append(list(vals.values()))
        res = np.array(features)
        return res.astype(float)

def canonize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)

dataset_dir = '../data'   # Change this to your dataset directory

train_df_round1 = pd.read_csv(f'{dataset_dir}/round1_train_data.csv')
train_df_round2 = pd.read_csv(f'{dataset_dir}/round2_train_data.csv')

train_df_round1['rxntype'] = 1

train_df = pd.concat([train_df_round1, train_df_round2])

print(f'Training set size: {len(train_df)}')

grouped=train_df.groupby('rxntype')
train_df_dict = {rxntype: group for rxntype, group in grouped}

columns=['Reactant1','Reactant2','Product','Additive','Solvent']
target_columns = ['Yield'] # list of names of the columns containing targets

n_rxn=8

# fp=MordredFingerprint(n_jobs=-1)
fp=featurizers_rdkit()

columns=['Reactant1','Reactant2','Product','Additive','Solvent']
icol=0
for col in columns:
    print(f"Treating {col}")
    smiles=train_df[col].tolist()
    smi_set=set()
    smi_dict={}
    for smi in smiles[:]:
        smi=canonize(smi)
        smislist=[smi]
        if smi.count('.')>0:
            smislist=smi.split('.')
        for itemsmi in smislist:
            smi_set.add(itemsmi)

    smi_list=list(smi_set)
    features=fp.transform(smi_list)
    for i in range(len(smi_list)):
        smi_dict[smi_list[i]]=features[i].tolist()

    savepath=r'../data/extra-mordred/'
    savepath=r'../data/extra-rdkit/'
    os.makedirs(savepath,exist_ok=True)
    with open(os.path.join(savepath,f"train-rdkitfeature-{col}.json"), 'w') as json_file:
        json.dump(smi_dict, json_file)

