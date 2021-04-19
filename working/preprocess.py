import pandas as pd
import numpy
from sklearn import preprocessing
import deepchem

def preprocess(dataframe):
    """
    Drops rows with missing values and converts categorical values to integers
    
    """
    
    # drops all rows with NaNs, although this can be modified to only look at rows in specific columns
    dataframe = dataframe.dropna() 
    
    # convert categorical values, if any exist, into integers
    for column in dataframe:
        if dataframe[column].dtype == object:
            l1 = preprocessing.LabelEncoder()
            l1.fit(dataframe[column])
            dataframe[column] = l1.transform(dataframe[column])
        else:
            continue

    return dataframe

def df_to_dataset(dataframe, X, y, w, ids):
    """
    Converts dataframe to DeepChem dataset object
    """
    
    to_dataset = deepchem.data.NumpyDataset(X, y, w, ids)
    dataset = to_dataset.from_dataframe(dataframe)
    
    return dataset

