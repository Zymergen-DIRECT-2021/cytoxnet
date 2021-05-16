### TODO ###
# Add functionallity to remove 'id' column before adding new one
# Write tests
# Make sure returned CSV saved to same directory as input CSV


import pandas as pd


def merge_joined_columns(dataframe):
    """
    To perform a full-outer join of two tables using SQLite, a union of two
    left-outer joins is usually used instead.  This can have the unwanted
    effect of creating multiple instances of the column on which the joins
    were made.  For example, if such a full-outer join is performed with two 
    tables on the column 'smiles' and the 'smiles' column is included in the 
    SELECT funtion, the resulting table will have two 'smiles' columns. 
    This function takes in such a table in the form of a dataframe and merges 
    these columns into a single column, outputting a dataframe with only one 
    merged column.

    Parameters
    ----------
    dataframe: pandas dataframe representing a table with only multiple instances of one 
    column.

    Returns
    -------
    - dataframe: pandas dataframe where all columns have been 
    merged into a single column.
    - merged_column: name of the remaining column.

    """
    column_list = list(dataframe)
    
    for n in range(1, len(column_list)):
        first_column = dataframe.columns[0] # get name of first column
        next_column = dataframe.columns[n] # get name of next column
        dataframe[first_column] = dataframe[first_column].fillna(dataframe[next_column])
        
    # drop all but the first column
    # this will return a pandas series object, not a dataframe object
    dataframe = dataframe.iloc[:,0]

    # convert pandas series object to dataframe object
    dataframe = dataframe.to_frame()

    # get name of remaining column
    merged_column = dataframe.columns[0]
    
    return dataframe, merged_column


def create_ids(dataframe, merged_column):
    """
    Creates 'id' column populated by sequencial decending integer values 
    and returns modified dataframe.

    Parameters
    ----------
    - dataframe: pandas dataframe without an 'id' column.
    - merged_column: name of the column where nans should be dropped.

    Returns
    -------
    df: pandas dataframe with added 'id' column.

    """
    # drop all rows where the specified column is empty
    df = dataframe.dropna(axis=0, subset=[merged_column])

    # insert 'id' column at the first column position
    df.insert(0, 'id', range(1, 1 + len(df)))
    
    return df


def modify_joined_table(csv_file):
    """
    Wrapping function for taking in a CSV file that has been generated
    from unioins of multiple left-outer joins and returning a CSV after 
    merging duplicate columns and adding a new 'id' column.

    Parameters

    """
    dataframe = pd.read_csv(csv_file)
    merged_df = merge_joined_columns(dataframe)
    ids_df = create_ids(merged_df)
    ids_df.to_csv('new_ids.csv', index=False)
    
    return