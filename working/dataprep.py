import pandas as pd
import numpy as np
from sklearn import preprocessing
import deepchem as dc

def convert_to_categorical(dataframe, cols=None):
    """
    Converts non-numerical categorical values to integers.  This function is most useful for ordinal variables.
    
    Parameters
    ----------
    - dataframe: featurized dataframe
    - cols: list of categorical columns to convert to integers
    
    Returns
    -------
    - modified dataframe with user selected columns with categorical string values converted to integer values
    """
    
    # generate binary values using get_dummies
    if cols is not None:
        if type(cols) == str: # allow single columns to be input as strings
            cols = [cols]
        else:
            pass
        
        for col in cols:
            if len(df1[col].shape) == 1: # for 1D arrays (usally y, target variables)
                # define label encoder
                encoder = preprocessing.LabelEncoder()
                # create new columns and preserve the original columns
                dataframe.loc[:,col+'_encoded'] = encoder.fit_transform(dataframe[col])
            else: # for 2D arrays (usually X, input features)
                # define ordinal encoder
                encoder = preprocessing.LabelEncoder()
                # create new columns and preserve the original columns
                dataframe.loc[:,col+'_encoded'] = encoder.fit_transform(dataframe[col])
        
    else:
        pass
    
    return dataframe


def convert_to_dataset(dataframe, 
                       X_col: str = 'X', 
                       y_col: str = 'y', 
                       w_col: str = None ,
                       id_col: str = None, 
                       return_csv = False):
    """
    Converts dataframe into a deepchem dataset object.
    
    Parameters
    ----------
    - dataframe: featurized dataframe
    - X_col: (str or List[str]) name(s) of the column(s) containing the X array.
    - y_col: (str or List[str]) name(s) of the column(s) containing the y array.
    - w_col: (str or List[str]) name(s) of the column(s) containing the w array.
    - id_col: (str) name of the column containing the ids.
    - return_csv: (True/False) whether a viewable csv of the data will be returned with the dataset object.

    Returns
    -------
    - deepchem dataset
    - (conditional) csv of dataframe
    """
    # define x
    X = dataframe[X_col]
    
    # define y
    y = dataframe[y_col]
    
    # define weight
    if w_col is not None:
        w = dataframe[w_col]
    else:
        w = None
    
    # define ids
    if id_col is not None:
        ids = dataframe[id_col]
    else:
        ids = None
    
    # create deepchem dataset object
    to_dataset = dc.data.NumpyDataset(X, y, w, ids)
    dataset = to_dataset.from_dataframe(dataframe)
    
    # return dataset and csv of current dataframe if return_csv equals True
    # return only dataset if return_csv equals False or unspecified
    if return_csv is True:
        csv = dataframe.to_csv(index = True, header=True)
        return dataset, csv
    else:
        return dataset

def data_transformation(dataset, 
                        transformations: list = ['NormalizationTransformer'], 
                        to_transform: list = [],  
                        **kwargs):
    """
    Transforms and splits deepchem dataset

    Parameters
    ----------
    - dataset: deepchem dataset to be transformed
    - transformations: (List[str]) list of transformation methods to pass dataset through
    - to_transform: (list[str]) list of elements to transform, and can include 'X', 'y', or 'w'
    - **kwargs: keyword arguments to be passed to the selected transformer

    Returns
    -------
    Transformed dataset
    """

    # feed dataset into list of transformers sequentially, returning a single transformed dataset
    for transformation in transformations:
        if to_transform is not None:
        
            if(all(elem in to_transform for elem in ['X', 'y', 'w'])):
                transformer = getattr(dc.trans, transformation)(transform_X=True, transform_y=True, transform_w=True, dataset=dataset, **kwargs)
            elif(all(elem in to_transform for elem in ['X', 'y'])):
                transformer = getattr(dc.trans, transformation)(transform_X=True, transform_y=True, dataset=dataset, **kwargs)
            elif 'X' in to_transform:
                transformer = getattr(dc.trans, transformation)(transform_X=True, dataset=dataset, **kwargs)
            elif 'y' in to_transform:
                transformer = getattr(dc.trans, transformation)(transform_y=True, dataset=dataset, **kwargs)
            else:
                transformer = getattr(dc.trans, transformation)(dataset=dataset, **kwargs)
        else:
            transformer = getattr(dc.trans, transformation)(dataset=dataset, **kwargs)
            
        dataset = transformer.transform(dataset)
    
    return dataset


"""
This is a version to use which works with the RandomSplitter 
that we can use for testing while I fix the bugs in the correct version.
"""

def data_splitting(dataset,  
                   split_type: str = 'train_valid_test_split', 
                   k: int = None, 
                   frac_train: float = None, 
                   frac_valid: float = None, 
                   frac_test: float = None, 
                   log_every_n: int = None):
    """
    Transforms and splits deepchem dataset

    Parameters
    ----------
    - dataset: deepchem dataset to be split
    - splitter: (str) class of deepchem split method
    - split_type: (str) type of split (k_fold_split/train_test_split/train_valid_test_split)
    - k: int
    - frac_train: float 
    - frac_valid: float 
    - frac_test: float 
    - log_every_n: int

    Returns
    -------
    Split dataset
    """

    # split data
    #data_splitter = getattr(dc.splits, splitter)
    data_splitter = dc.splits.RandomSplitter()

    # this only allows the following three split_types to be used
    # the 'split' option is excluded since it seems to do the same thing as 'train_valid_test_split' but returns non-dataset objects
    
    if split_type == 'k_fold_split' or split_type == 'k':
        data_split = data_splitter.k_fold_split(dataset=dataset, k=k)
    elif split_type == 'train_test_split' or split_type == 'tt':
        data_split = data_splitter.train_test_split(dataset=dataset, frac_train=frac_train)
    elif split_type == 'train_valid_test_split' or split_type =='tvt':
        data_split = data_splitter.train_valid_test_split(dataset=dataset, frac_train=frac_train, frac_valid=frac_valid, frac_test=frac_test)
    elif split_type == 'generate_scaffolds':
        if hasattr(data_splitter, 'generate_scaffolds'):
            data_split = data_splitter.generate_scaffolds(dataset, log_every_n) # Unsure about functionality, code may need to be added
        else:
            raise AttributeError ('split_type may only be set as generate_scaffolds if splitter set as ScaffoldSplitter')
    else: 
        print('split_type string is not a recognized split') # should change this to raise an error of some kind
    
    return data_split