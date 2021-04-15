"""
Input/Output module for loading data and generating output files.  
"""

import pandas as pd


def load_data(datafile, columns='all'):
    """
    Loads a data file into a dataframe containing the raw data.
    
    Parameters
    ----------
    datafile: the file containing the data to be loaded 
    columns: the column (names?numbers?either?) that the user wants to load,
        default='all'
    
    Returns
    -------
    DataFrame containg the raw data from the csv

    """
    # assert file exists and contains data
    # run any more checks specific to the data that we may want to add
    # support for multiple file types?
    # load a csv file into a dataframe
    df = pd.load_csv(datafile)
    return df




