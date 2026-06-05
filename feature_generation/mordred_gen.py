from skfp.fingerprints import MACCSFingerprint, MordredFingerprint, EStateFingerprint, AutocorrFingerprint, GhoseCrippenFingerprint, MQNsFingerprint, LaggnerFingerprint
from rdkit.Chem import Descriptors
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger,Chem
import numpy as np
import os
RDLogger.DisableLog('rdApp.*')
#定义了 featurizers_rdkit 类，用于将 SMILES 字符串转换为分子描述符特征。针对每个反应类型（rxntype），对 Reactant1、Reactant2、Product、Additive、Solvent 列的 SMILES 字符串进行特征提取，并将这些特征按列拼接。

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

class featurizers_rdkit():
    def transform(self,smis):
        features=[]
        for smi in smis:
            vals = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smi))
            features.append(list(vals.values()))
        res = np.array(features)
        return res.astype(float)

# fp=MordredFingerprint(n_jobs=-1)1
fp=featurizers_rdkit()

train_x_all=[]
train_y_all=[]
rxn_indx=[0]
for j in range(1,n_rxn+1):    
    rxn_x=[]
    for col in columns:
        smile_list=train_df_dict[j][col]
        col_fp=fp.transform(smile_list[:])           
        rxn_x.append(col_fp)
    rxn_x=np.concatenate(rxn_x,axis=1)
    train_x_all.append(rxn_x)
    train_y_all.append(train_df_dict[j]['Yield'].values[:])
    rxn_indx.append(rxn_indx[-1]+len(rxn_x))
# train_x_all = np.concatenate(train_x_all,axis=0)
  
# nonzero_columns=np.where(np.var(train_x_all,axis=0)  != 0)
# print("Non zero bits:",len(nonzero_columns[0]))
# train_x_all=train_x_all[:, np.var(train_x_all, axis=0) != 0]
# train_x_all = [train_x_all[rxn_indx[i]:rxn_indx[i+1]] for i in range(len(rxn_indx) - 1)]

savepath=r'../data/extra-mordred/'
savepath=r'../data/extra-rdkit/'

os.makedirs(savepath,exist_ok=True)
for i in range(len(train_x_all)):
    print(f"Feature shape of rxn {i+1}: {train_x_all[i].shape}")  
    # np.savetxt(os.path.join(savepath,f"train-mordedfeature-rxn{i+1}.gz"), train_x_all[i], delimiter=",")
    # np.savetxt(os.path.join(savepath,f"train-target-rxn{i+1}.gz"), train_y_all[i], delimiter=",")
    np.savetxt(os.path.join(savepath,f"train-rdkitfeature-rxn{i+1}.gz"), train_x_all[i], delimiter=",")
    np.savetxt(os.path.join(savepath,f"train-target-rxn{i+1}.gz"), train_y_all[i], delimiter=",")

