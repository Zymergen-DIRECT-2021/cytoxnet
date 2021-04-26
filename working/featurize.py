import pandas as pd
import data_load_clean as dlc 
import rdkit
from rdkit import Chem
import deepchem as dc
import numpy as np


df = dlc.load_chembl()

def smiles_to_mol(dataframe):
    mols = []
    for smiles in df['Smiles']:
        mol =  Chem.MolFromSmiles(smiles)
        mols.append(mol)
    df['Mol'] = mols
    return df

df_1 = smiles_to_mol(df)

def add_circular_fingerprint(df):
    featurizer = dc.feat.CircularFingerprint()
    f_list = [] 
    for mol in df['Mol']:
        f = featurizer.featurize(mol)
        f_list.append(f)
    df['CircularFingerprint'] = f_list
    return df

df_2 = add_circular_fingerprint(df_1)
print(df_2)