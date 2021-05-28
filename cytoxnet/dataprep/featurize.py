import os
import ast
import re

from rdkit import Chem
import deepchem as dc
import numpy as np
import pandas as pd

import cytoxnet.dataprep.dataprep as dp


def molstr_to_Mol(dataframe, id_col='InChI String'):
    """
    Converts DataFrame column containing the string representations of
    molecules and add a corresponding column of Mol objects.

    Parameters
    ----------
    dataframe: DataFrame containing a column with string representations
    of molecules
    id_col: label for the column containg the string representations for
    converstion, default='InChI String'- can be changed for the user's
    standard column name

    Returns
    -------
    DataFrame with additional column containing Mol objects for each molecule,
    column label is 'Mol'

    """
    dataframe = dataframe.copy()
    mols = []
    if 'inchi' in id_col.lower():
        for inchi in dataframe[id_col]:
            mol = Chem.MolFromInchi(inchi)
            mols.append(mol)

    elif 'smiles' in id_col.lower():
        for smiles in dataframe[id_col]:
            mol = Chem.MolFromSmiles(smiles)
            mols.append(mol)

    dataframe['Mol'] = mols
    return dataframe


def from_np_array(array_string):
    """Convert a string to numpy array.

    Used for loading string arrays in pandas dataframes.
    """
    try:
        array_string = re.sub('\\[\\s*', '[', array_string)
        array_string = ','.join(array_string.split())
        out = np.array(ast.literal_eval(array_string))
    except BaseException:
        out = None
    return out


def add_features(dataframe,
                 id_col='smiles',
                 method='CircularFingerprint',
                 codex=None,
                 canonicalize=True,
                 drop_na=True,
                 **kwargs):
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
    id_col: label of the column containing the Mol objects, default based
    on function for adding a Mol column
    codex : str
        path to the codex containing smiles and features
    canonicalize : bool
        Whether or not to first canonicalize the id_column
    drop_na : bool
        Whether to drop nas of post featurization data

    Returns
    -------
    DataFrame containing a column of the featurized representation of the Mol
    object with the featurization method as the column ID

    """
    dataframe = dataframe.copy()
    if canonicalize:
        dataframe[id_col] = dataframe[id_col].apply(
            lambda x: dp.canonicalize_smiles(x)
        )
    # try to get features from codex
    if codex is not None:
        assert os.path.exists(codex)
        compounds = pd.read_csv(
            codex, index_col=0, converters={method: from_np_array}
        )
        if method in compounds.columns:
            # determine which ids are in the codex
            dataframe['ind'] = dataframe.index
            overlap = dataframe.merge(
                compounds, how='inner', on=[id_col]
            ).set_index('ind', drop=True)
            dataframe[method] = overlap[method]
            dataframe.drop(columns=['ind'], inplace=True)
        else:
            dataframe[method] = None
    else:
        dataframe[method] = None

    # compute features for the non featurized ones
    dataframe_ = dataframe[dataframe.isna()[method]]
    dataframe_ = molstr_to_Mol(dataframe_, id_col=id_col)

    featurizer = getattr(dc.feat, method)(**kwargs)
    f = list(featurizer.featurize(dataframe_['Mol'].values))
    dataframe_[method] = f
    dataframe.loc[dataframe_.index, method] = dataframe_[method]

    # drop na
    if drop_na:
        # raw nans
        dataframe.dropna(subset=[method], inplace=True)
        # nans within the arrays
        featarray = np.vstack(dataframe[method].values)
        if featarray.dtype == np.object:
            pass
        else:
            naninds = np.unique(np.where(np.isnan(featarray))[0])
            dataframe.drop(index=dataframe.index[naninds], inplace=True)
    return dataframe
