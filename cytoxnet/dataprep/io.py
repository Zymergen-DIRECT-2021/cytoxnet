"""
Input/Output module for loading data and generating output files.
"""
import os
import pkg_resources

import pandas as pd

import cytoxnet.dataprep.featurize as ft


def load_data(csv_file,
              cols=None,
              id_cols=None,
              duplicates='drop',
              nans='keep'):
    """
    Loads a data file into a dataframe containing the raw data.

    Parameters
    ----------
    datafile: the file containing the data to be loaded
    cols: Columns in the csv to keep, default None (keep all)
    id_cols: the column (names?numbers?either?) that the user wants to
        use to handle duplicates and nana(ie remove any duplicates based on
        the inputted column id) or a list of them, default=None
    duplicates: how to handle duplicates in the id_cols. Options:
        'drop' - drop any duplicates, retaining the first instance
        'keep' - keep duplcates
    nans: how to handle nans in the id_cols. Options:
        'drop' - drop any nans
        'keep' - keep nans

    Returns
    -------
    DataFrame containg the raw data from the csv,
    with duplicates and nans handled if a column for doing so
    is passed

    """
    if isinstance(id_cols, str):
        id_cols = [id_cols]
    # assert file exists and contains data
    if isinstance(csv_file, str):
        assert os.path.exists(csv_file), 'File name does not exist'
    # run any more checks specific to the data that we may want to add
    # load a csv file into a dataframe
    df = pd.read_csv(csv_file, index_col=0)
    # drop any unwanted columns
    if cols is not None:
        df = df[cols]
    else:
        pass

    # handle dups and nans
    if nans == 'drop':
        df.dropna(subset=cols, inplace=True)
    elif nans == 'keep':
        pass
    else:
        raise ValueError(
            '{} not a valid option for nans'.format(nans)
        )
    if id_cols is not None:
        if duplicates == 'drop':
            df.drop_duplicates(subset=id_cols, inplace=True)
        elif duplicates == 'keep':
            pass
        else:
            raise ValueError(
                '{} not a valid option for duplicate'.format(duplicates)
            )

    else:
        pass
    return df


def load_chembl_ecoli():
    # get the path in the package
    path = pkg_resources.resource_stream(
        __name__, '../data/chembl_ecoli_MIC.csv'
    )
    df = load_data(path,
                   cols=['smiles', 'MIC'],
                   id_cols=['smiles'],
                   nans="drop")
    return df


def load_zhu_rat():
    path = pkg_resources.resource_stream(
        __name__, '../data/zhu_rat_LD50.csv'
    )
    df = load_data(path,
                   cols=['smiles', 'LD50'],
                   id_cols=['smiles'],
                   nans='drop')
    return df


def load_lunghini(species=['algea', 'fish', 'daphnia'], nans='drop'):
    assert len(species) > 0,\
        "Secies must be one or more of algea, fish, daphnia"

    path = pkg_resources.resource_stream(__name__, '../data/lunghini.csv')
    cols = ['smiles']
    # get the target columns to consider
    if 'algea' in species:
        cols.append('algea_EC50')
    if 'fish' in species:
        cols.append('fish_LC50')
    if 'daphnia' in species:
        cols.append('daphnia_EC50')

    df = load_data(path,
                   cols=cols,
                   id_cols=['smiles'],
                   nans=nans)
    return df

def create_data_codex(path='./data_codex.csv',
                      featurizer=None,
                      **kwargs):
    lunghini = load_lunghini(nans='keep')
    zhu_rat = load_zhu_rat()
#     chembl_ecoli = load_chembl_ecoli()
    
    codex = pd.DataFrame(pd.concat(
        [
            lunghini['smiles'],
            zhu_rat['smiles'],
#             chembl_ecoli['smiles']
        ],
        ignore_index=True
    ))
    
    codex.drop_duplicates(subset='smiles', inplace=True)
    
    if featurizer is not None:
        assert all([type(f) == str for f in featurizer]),\
            "featurizer should be a list of featurizers to use"
        codex = ft.molstr_to_Mol(codex, strcolumnID='smiles')
        for f in featurizer:
            codex = ft.add_features(codex,
                                    MolcolumnID='Mol',
                                    method=f,
                                    **kwargs)
        codex.drop('Mol', axis=1, inplace=True)
    codex.to_csv(path)
    return

def add_dataset(dataframe,
                id_col='smiles',
                path = './data_codex.csv',
                new_featurizer=None,
                **kwargs):
    assert id_col in dataframe.columns, "dataframe should have `id_col`"
    master = load_data(path, nans='keep', duplicates='keep')
    dataframe_ = dataframe.copy()
    dataframe_.rename(
        columns={id_col: 'smiles'}, inplace=True
    )
    
    # first extract the non duplicate molecules
    common = dataframe_.merge(master, on='smiles')['smiles']
    uniques = pd.DataFrame(dataframe_[
        ~dataframe_['smiles'].isin(common)
    ]['smiles'])
    
    # create a mol object for these smiles
    uniques = ft.molstr_to_Mol(uniques, strcolumnID='smiles')
    
    # compute the features already in the codex for these unique values
    for col_name in master.columns:
        if col_name != 'smiles':
            uniques = ft.add_features(uniques,
                                      MolcolumnID='Mol',
                                      method=col_name,
                                      **kwargs)

    # add these new values to the codex
    master = pd.concat([master, uniques], ignore_index=True)
    
    if new_featurizer is not None:
        assert all([type(f) == str for f in new_featurizer]),\
            "new_featurizer should be a list of featurizers to use"
        master = ft.molstr_to_Mol(master, strcolumnID='smiles')
        for f in new_featurizer:
            master = ft.add_features(master,
                                     MolcolumnID='Mol',
                                     method=f)
    # drop Mol
    master.drop('Mol', axis=1, inplace=True)
    print(path)
    master.to_csv(path)
    return