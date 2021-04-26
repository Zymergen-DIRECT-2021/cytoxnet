import pandas as pd

def load_chembl():
    # need to change file path to where the data is on your local for now
    # later can decide what, if any, data we might want to put on github 
    csv_file = '/Users/nida/Documents/ChEMBL_Example - Sheet1.csv'
    df = pd.read_csv(csv_file)
    df_1 = df.drop_duplicates(subset='Molecule ChEMBL ID')
    df_2 = df_1.reset_index()
    return df_2
