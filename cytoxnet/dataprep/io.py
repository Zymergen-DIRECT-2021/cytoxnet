"""
Input/Output module for loading data and generating output files.
"""
import os

import numpy as np
import pandas as pd
import deepchem as dc

import cytoxnet.dataprep.featurize as ft
import cytoxnet.dataprep.dataprep as dp
import cytoxnet.data as data

np.set_printoptions(threshold=np.inf)


def load_data(file,
              cols=None,
              id_cols=None,
              duplicates='drop',
              nans='keep'):
    """
    Loads a data file into a dataframe containing the raw data.

    Parameters
    ----------
    datafile: the file containing the data to be loaded
    cols: Columns in the csv to drop, default None (keep all)
    id_cols: the column (names?numbers?either?) that the user wants to
        use to handle duplicates and nans(ie remove any duplicates based on
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
    if isinstance(file, str):
        if os.path.exists(file):
            pass
        # or is in the package data
        else:
            file_ = None
            package_data_location = data.__path__._path[0]
            data_files = os.listdir(package_data_location)
            for f in data_files:
                if file == f.split('.')[0]:
                    file_ = package_data_location + '/' + f
                    break
            if file_ is None:
                raise FileNotFoundError(
                    'file must be a valid path or the name without extension\
 of a file in the package data.'
                )
            else:
                file = file_

    # run any more checks specific to the data that we may want to add
    # load a csv file into a dataframe
    df = pd.read_csv(file, index_col=0)
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


def create_compound_codex(db_path='./database',
                          id_col='smiles',
                          featurizers=None,
                          **kwargs):
    """
    Create a compound codex for a combined database.

    Creates a master csv file that tracks the unique canonicalized smiles of
    all data in the database, and stores features for those data.

    Parameters
    ----------
    db_path : str
        The path to the folder to contain database files. Will create direcory
        if it does not exist.
    id_col : str
        The column in all dataframes representing the compound id.
    featurizers : str or list of str
        The featurizer/s to initialize the compounds codex with.
    """
    if featurizers is not None:
        if not isinstance(featurizers, list):
            featurizers = [featurizers]
        assert all([hasattr(dc.feat, f) for f in featurizers]),\
            "featurizer should be a list of valid featurizers to use"
        master = pd.DataFrame(columns=[id_col, *featurizers])
    else:
        master = pd.DataFrame(columns=[id_col])
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    master.to_csv(db_path + '/compounds.csv')
    return


def add_datasets(dataframes,
                 names,
                 id_col='smiles',
                 db_path='./database',
                 new_featurizers=None,
                 **kwargs):
    """Add a new set of data to the tracked database.

    Update the compounds csv with new dataset/s canonicalized, and saves
    csvs to the database folder with foreign keys tracked.

    Parameters
    ----------
    dataframes : dataframe or string or list of them
        The datasets to add. If it is a string object, we will attempt to load
        the file at the string path or a file in the package data.
    names : str or list of str
        Names of the datasets passed. 
    id_col : str
        The column in all dataframes representing the compound id
    db_path : str
        The path to the folder containing database files.
    new_featurizers : str or list of str, default None
        Featurizer names to apply to the new data as well as all current data.
    """
    # get data from package if it is not already in dataframe form
    if not isinstance(dataframes, list):
        dataframes = [dataframes]
    if not isinstance(names, list):
        names = [names]
    assert len(names) == len(dataframes),\
        'names should be the names of the datasets passed, with the same len'

    dataframes_ = []
    for df in dataframes:
        if isinstance(df, pd.core.frame.DataFrame):
            dataframes_.append(df.copy())
        elif isinstance(df, str):
            try:
                loaded_df = load_data(df)
                dataframes_.append(loaded_df)
            except BaseException:
                raise ValueError(
                    f'One of the dataframes passed ({df}) was a string, but\
 does not correspond to a file or package dataset.'
                )
        else:
            raise ValueError(
                f'Could not add input dataset {df} fo type {type(df)}.'
            )
        assert id_col in dataframes_[-1].columns,\
            f'Cannot add a dataset that does not have the id_col={id_col}\
 column'

    master = pd.read_csv(db_path + '/compounds.csv', index_col=0)
    assert id_col in master.columns, f'The master data file should have the\
 column id_col=`{id_col}`'

    dfs_out = []
    for i, df in enumerate(dataframes_):

        # canonicalize
        df[id_col] = df[id_col].apply(lambda x: dp.canonicalize_smiles(x))
        df.dropna(subset=[id_col], inplace=True)
        # first extract the molecules that already exist in the codex
        common = df.merge(master, on=id_col)[id_col]
        # and now the ones that do not - the data may have duplicates and we
        # only want to one set
        uniques = pd.DataFrame(df[
            ~df[id_col].isin(common)
        ][[id_col]])
        uniques.drop_duplicates(subset=[id_col], inplace=True)
        # compute the features already in the codex for these unique values
        for col_name in master.columns:
            if col_name != id_col:
                uniques = ft.add_features(uniques,
                                          id_col=id_col,
                                          method=col_name,
                                          **kwargs)
        # add these new values to the codex
        master = pd.concat([master, uniques], ignore_index=True)
        # get the foreign key for the data
        fkeys = []
        for sm in df[id_col].values:
            fkey = int(master.index[master[id_col] == sm].values)
            fkeys.append(fkey)
        df['foreign_key'] = fkeys
        dfs_out.append(df)
        df.to_csv(db_path + '/' + names[i] + '.csv')

    if new_featurizers is not None:
        if not isinstance(new_featurizers, list):
            new_featurizers = [new_featurizers]
        assert all([isinstance(f, str) for f in new_featurizers]),\
            "new_featurizers should be a list of featurizers to use"
        for f in new_featurizers:
            master = ft.add_features(master,
                                     id_col=id_col,
                                     method=f)
    master.to_csv(db_path + '/compounds.csv')
    cleaned_db_frames = dict(zip(names, dfs_out))
    return cleaned_db_frames
