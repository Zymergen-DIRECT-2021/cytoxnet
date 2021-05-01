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
              nans = 'drop'):
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
    if id_cols is not None:
        if duplicates == 'drop':
            df.drop_duplicates(subset=id_cols, inplace=True)
        elif duplicates == 'keep':
            pass
        else:
            raise ValueError(
                '{} not a valid option for duplicate'.format(duplicates)
            )
            
        if nans == 'drop':
            df.dropna(subset=id_cols, inplace=True)
        elif nans == 'keep':
            pass
        else:
            raise ValueError(
                '{} not a valid option for nans'.format(nans)
            )
    else:
        pass
    return df

def load_chembl_ecoli():
    # get the path in the package
    path = pkg_resources.resource_stream(__name__, '../data/chembl_ecoli_MIC.csv')
    df = load_data(path,
                   cols=['smiles', 'MIC'],
                   id_cols=['smiles', 'MIC'])
    return df
    