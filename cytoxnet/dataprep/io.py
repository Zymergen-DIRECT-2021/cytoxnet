"""
Input/Output module for loading data and generating output files.
"""
import os
import pkg_resources

import pandas as pd


def load_data(csv_file,
              cols = None,
              id_cols = None,
              duplicates = 'drop',
              nans = 'keep'):
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
    if type(id_cols) == str:
        id_cols = [id_cols]
    # assert file exists and contains data
    if type(csv_file) == str:
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
                   id_cols=['smiles'])
    return df
    
def load_zhu_rat():
    path = pkg_resources.resource_stream(
        __name__, '../data/zhu_rat_LD50.csv'
    )
    df = load_data(path,
                   cols=['smiles', 'LD50'],
                   id_cols=['smiles'])
    return df

def load_fillipo(species=['algea', 'fish', 'daphnia'], nans = 'drop'):
    assert len(species) > 0,\
        "Secies must be one or more of algea, fish, daphnia"
        
    path = pkg_resources.resource_stream(__name__, '../data/fillipo.csv')
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
        