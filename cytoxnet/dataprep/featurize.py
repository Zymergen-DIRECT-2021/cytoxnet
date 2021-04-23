import pandas as pd
import rdkit
from rdkit import Chem
import deepchem as dc
import numpy as np

def molstr_to_Mol(dataframe, strcolumnID='InChI String'):
    """
    Converts DataFrame column containing the string representations of
    molecules and add a corresponding column of Mol objects.

    Parameters
    ----------
    dataframe: DataFrame containing a column with string representations
    of molecules
    strcolumnID: label for the column containg the string representations for
    converstion, default='InChI String'- can be changed for the user's
    standard column name

    Returns
    -------
    DataFrame with additional column containing Mol objects for each molecule,
    column label is 'Mol'

    """
    mols = []
    if 'inchi' in strcolumnID.lower():
        print('incho')
        for inchi in dataframe[strcolumnID]:
            mol = Chem.MolFromInchi(inchi)
            mols.append(mol) 

    elif 'smiles' in strcolumnID.lower():
        print('smiles')
        for smiles in dataframe[strcolumnID]:
            mol =  Chem.MolFromSmiles(smiles)
            mols.append(mol)

    dataframe['Mol'] = mols
    return dataframe

def add_features(dataframe, MolcolumnID='Mol', method='CircularFingerprint'):
    """
    Featurizes a set of Mol objects using the desired feturization method.

    note: may want to change setup of parameters here
    note: may be better suited as a class
    note: might not need separate featurizer for graph featurization-
    could include here I think

    list of possible featurizers to wrap up here:
    https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html

    Parameters
    ----------
    dataframe: a DataFrame containing a column with Mol objects-
    may want to play around with how we want to input the set of Mols
    for featurization
    columnID: label of the column containing the Mol objects, default based
    on function for adding a Mol column

    Returns
    -------
    DataFrame containing a column of the featurized representation of the Mol
    object with the featurization method as the column ID

    """
    # Check that set contains Mol objects
    assert isinstance(dataframe['Mol'][0], rdkit.Chem.Mol), 'Mol column does not contain Mol object'
    featurizer = getattr(dc.feat, method)()
    f_list = []
    for mol in dataframe['Mol']:
        f = featurizer.featurize(mol)
        f_list.append(f)
    dataframe[method] = f_list

    # assert isinstance(object_in_set, rdkit.Chem.rdchem.Mol)

    # molecules format optins-  rdkit.Chem.rdchem.Mol /
    # SMILES string / iterable
    # either a loop or pass an iterable set of Mol
    # iterable = convert column of Mol to array
    # features = featurizer.featurize(iterable)
    # add the featurized representation into the passed dataframe
    # dataframe[method] = features

    return dataframe

def add_circular_fingerprint(df):
    featurizer = dc.feat.CircularFingerprint()
    f_list = [] 
    for mol in df['Mol']:
        f = featurizer.featurize(mol)
        f_list.append(f)
    df['CircularFingerprint'] = f_list
    return df

def get_descriptors(dataframe, descriptor_type):
    """
    Extracts molecular descriptors and adds them to a new column
    Something to look into in the future
    What descriptors might we want? how can we get them?
    will need a featurizing function that can take any descriptors we
    extract here
    """
    # dataframe[descriptor_type] = descriptorx_list
    return dataframe
