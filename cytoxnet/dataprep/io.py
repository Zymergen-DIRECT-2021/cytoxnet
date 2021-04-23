"""
Input/Output module for loading data and generating output files.
"""

import pandas as pd
import os


def load_data(csv_file, col_id=None):
    """
    Loads a data file into a dataframe containing the raw data.

    Parameters
    ----------
    datafile: the file containing the data to be loaded
    col_id: the column (names?numbers?either?) that the user wants to
        use to remove duplicates (ie remove any duplicates based on
        the inputted column id), default=None

    Returns
    -------
    DataFrame containg the raw data from the csv,
    with duplicates removed if a column for doing so
    is passed

    """
    # assert file exists and contains data
    assert os.path.exists(csv_file), 'File name does not exist'
    # run any more checks specific to the data that we may want to add
    # load a csv file into a dataframe
    df = pd.read_csv(csv_file)
    if col_id is not None:
        df_1 = df.drop_duplicates(subset=col_id)
    else:
        df_1 = df
    df_2 = df_1.reset_index()
    # remove/ignore unwanted columns?
    return df_2
