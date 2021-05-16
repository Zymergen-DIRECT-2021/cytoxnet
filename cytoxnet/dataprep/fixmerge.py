import pandas as pd



def merge_joined_columns(dataframe):
    
    column_list = list(dataframe)
    
    for n in range(1, len(column_list)):
        first_column = dataframe.columns[0] # Get name of first column
        next_column = dataframe.columns[n] # Get name of next column
        dataframe[first_column] = dataframe[first_column].fillna(dataframe[next_column])
        
    dataframe = dataframe.iloc[:,0] # Drop all but the first column
    
    return dataframe


def create_ids(dataframe):

    if isinstance(dataframe, pd.core.series.Series) is True:
        df = dataframe.to_frame()
    else:
        df = dataframe
    
    df_dropna = df.dropna(axis=0, subset=['smiles'])    
    df_dropna.insert(0, 'id', range(1, 1 + len(df_dropna)))
    
    return df_dropna


def output_csv(dataframe):
    dataframe.to_csv('new_ids.csv', index=False)
    return


def modify_joined_table(csv_file):
    dataframe = pd.read_csv(csv_file)
    merged_df = merge_joined_columns(dataframe)
    ids_df = create_ids(merged_df)
    output_csv(ids_df)
    
    return