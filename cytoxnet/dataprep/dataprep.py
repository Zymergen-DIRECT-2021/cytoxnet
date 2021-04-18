## potentially useful: deepchem dataset class or pandas split
## having our data in the deepchem.dataset form at the end is very helpful


"""
Data splitting module for creating train/test splits  
"""

#import pandas as pd
#import deepchem
#import sklearn


#def convert_cats(dataframe):
    """
    Drops rows with missing values and converts categorical values to integers.
    
    Parameters
    ----------
    dataframe: featurized dataframe
    
    Returns
    -------
    modified dataframe without any na values in the 'Mol' column 
    and all categorical columns with string values converted to integer values
    """
    
    ## Drop all na values in the 'Mol' column
    ## May want to expand the range of columns beyond 'Mol'
    
    #dataframe = dataframe.dropna()

    ## convert categorical values, if any exist, into integers
    ## might be able to use sklearn Label Encoder or OnHotEncoder
    ## might be necessary if we categorize target areas/tox methods/etc.
    ## Iterate through columns and convert any containing strings to ints
    
    #for column in dataframe:
    #  column_series_obj = dataframe[column]
    #	if dataframe[column].dtype not numeric:
    #    l = sklearn.preprocessing.LabelEncoder()
    #		l.fit(dataframe[column])
    #		dataframe.column = l.transform(dataframe.column)
    #	else continue
    #return dataframe






#def trans_data(dataframe, X, y, w, ids, 
#	transform_list: list = ['Normalize',], to_transform: str = 'X', **kwargs):
    
    """
    Converts featurized dataframe into a transformed dataset object.
    
    Parameters
    ----------
   	dataframe: featurized dataframe 
   	X: (str or List[str]) name(s) of the column(s) containing the X array.
   	y: (str or List[str]) name(s) of the column(s) containing the y array.
   	w: (str or List[str]) name(s) of the column(s) containing the w array.
   	ids: (str) name of the column containing the ids.
   	transform_list: (List[str]) list of transformation methods to pass dataset through
   	to_transform: (str) specification of whether to transform X, y, or w
   	**kwargs: key word arguments to be passed to selected transformer


    Returns
    -------
    transformed dataset object
    """
    
    ## convert dataframe to dataset using deepchem.data.NumpyDataset function
    
    #to_dataset = deepchem.data.NumpyDataset(X, y, w, ids)
	#dataset = to_dataset.from_dataframe(dataframe)

    #if to_transform == 'X':
    #	transform_X = True
    #    transform_y = False
   	#elif to_transform == 'y':
    #    transform_X = False
    #    transform_y = True

    ## feed dataset into list of transformers sequentially, returning a single transformed dataset
   	
    #for transform in transform_list:
    #   	transformer = getattr(deepchem.transformers, transform)(transform_X=transform_X, transform_y=transform_y, dataset=dataset **kwargs)
    #   	dataset = transformer.transform(dataset)

    #return dataset    








#def split_data(dataset, method='RandomSplitter', split_type='train_valid_test_split', 
#	k=None, frac_train=None, frac_valid=None, frac_test=None, seed=None, **kwargs):

    """
    Converts processed dataset into split dataset objects.
    
    Parameters
    ----------
   	dataset: dataset object to be split
   	method: (str) class of deepchem split method
   	split_type: (str) type of split (k_fold_split/train_test_split/train_valid_test_split)
   	k: (int) number of folds to split dataset into
   	frac_train: (float) fraction of data to be used for training split
   	frac_valid: (float) fraction of data to be used for validation split
   	frac_test: (float) fraction of data to be used for test split
   	seed: (int) random seed to use
   	**kwargs: keyward arguments specific to the input method

    
    Returns
    -------
    tuple or list of dataset objects
    """
    
    ## convert dataframe to dataset using deepchem.data.NumpyDataset function
    
    #data_splitter = getattr(deepchem.splits, method)(**kwargs)

    ## this only allows the following three split_types to be used
    ## the 'split' option is excluded since it seems to do the same thing as 'train_valid_test_split' but returns non-dataset objects
    ## might want to write code to reset certain defaults depending on which split_type is chosen (i.e. from None to 0.8)
    
    #if split_type == 'k_fold_split':
    #	data_split = data_splitter.split_type(dataset, k)
    #elif split_type == 'train_test_split':
    #	data_split = data_splitter.split_type(dataset, frac_train, seed)
    #elif split_type == 'train_valid_test_split':
    #	data_split = data_splitter.split_type(dataset, frac_train, frac_valid, frac_test, seed)
    #else: 
    #    raise error # might want to expand this so we can account for edge cases like use of the 'generate_scaffolds' function

    ## ensure k parameter is only being passed if all other parameters are set appropriately
    ## this should catch many (but not all) inappropriate input combinations
    
    #if k is not None:
    #	if split_type != 'k_fold_split':
    #		raise error # k should only be passed if using k_fold_split
    #	elif frac_train is not None:
    #		raise error # cannot have both be true
    #	elif frac_valid is not None:
    #		raise error # connot have both be true
    #	elif frac_test is not None:
    #		raise error # cannot have both be true
    #	elif seed is not None:
    #		raise error # cannot have both be true
    #	else continue 
    #else: 
    #    continue

    ## this will catch an additional inappropriate input combination not caught by previous if statement
    
    #if (((frac_valid is not None) or (frac_test is not None)) and split_type is not 'train_valid_test_split'):
    #	raise error # frac_valid and frac_test can only be passed if split_type is set to 'train_valid_test_split'
    #else: 
    #    continue

    #return data_split


