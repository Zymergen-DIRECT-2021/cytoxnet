"""
Data splitting module for creating train/test splits  
"""

#import pandas as pd
#import deepchem


#def convert_to_categorical(dataframe, cols: list of str):
  """
  Converts non-numerical categorical values to integers.
  
  Parameters
  ----------
  - dataframe: featurized dataframe
    
  Returns
  -------
  modified dataframe with user selected columns with categorical string values converted to integer values
  """

  # convert categorical string values within selected columns into integers
  # might be able to use sklearn Label Encoder, OnHotEncoder, or pandas dummies values
  # might be necessary if we categorize target areas/tox methods/etc.

  # generate binary values using get_dummies
#  dataframe = pd.get_dummies(dataframe, columns=cols)    

#  return dataframe


#def convert_to_dataset(dataframe, X_col: str = 'X', y_col: str = 'y', w_col: str = None, id_col: str = None):
	"""
	Converts dataframe into a deepchem dataset object.

	Parameters
	----------
	- dataframe: featurized dataframe
	- X_col: (str or List[str]) name(s) of the column(s) containing the X array.
  - y_col: (str or List[str]) name(s) of the column(s) containing the y array.
  - w_col: (str or List[str]) name(s) of the column(s) containing the w array.
  - id_col: (str) name of the column containing the ids.

  Returns
  -------
  deepchem dataset
	"""

#	to_dataset = dc.data.NumpyDataset(X, y, w, ids)
#  dataset = to_dataset.from_dataframe(dataframe)

#  return dataset

#def preprocess(dataset, transformations: list of str = ['NormTransform',], to_transform: str = 'X', splitter: str = 'RandomSplitter', split_type: 'train_valid_test_split', k = None, frac_train = None, frac_valid = None, frac_test = None, seed = None, **kwargs):

	"""
	Transforms and splits deepchem dataset

	Parameters
	----------
	- dataset: deepchem dataset to be transformed and/or split
	- transformations: (List[str]) list of transformation methods to pass dataset through
  - to_transform: (str) specification of whether to transform X, y, or w
  - method: (str) class of deepchem split method
  - split_type: (str) type of split (k_fold_split/train_test_split/train_valid_test_split)
  - k: (int) number of folds to split dataset into
  - frac_train: (float) fraction of data to be used for training split
  - frac_valid: (float) fraction of data to be used for validation split
  - frac_test: (float) fraction of data to be used for test split
  - seed: (int) random seed to use
  - **kwargs: keyword arguments to be passed to the selected transformer and splitter

  Returns
  -------
  Transformed and/or split ML ready dataset
	"""

#	if to_transform == 'X':
#		transform_X = True
#		transform_y = False
#	elif to_transform == 'y':
#		transform_X = False
#		transform_y = True

  # feed dataset into list of transformers sequentially, returning a single transformed dataset
   	
#  for transformation in transformations:
#  	transformer = getattr(deepchem.transformers, transformation)(transform_X=transform_X, transform_y=transform_y, dataset=dataset **kwargs)
#  	dataset = transformer.transform(dataset)

  # split data

#  data_splitter = getattr(deepchem.splits, method)(**kwargs)

  # this only allows the following three split_types to be used
  # the 'split' option is excluded since it seems to do the same thing as 'train_valid_test_split' but returns non-dataset objects
  # might want to write code to reset certain defaults depending on which split_type is chosen (i.e. from None to 0.8)
    
#  if split_type == 'k_fold_split':
#  	data_split = data_splitter.split_type(dataset, k)
#  elif split_type == 'train_test_split':
#  	data_split = data_splitter.split_type(dataset, frac_train, seed)
#  elif split_type == 'train_valid_test_split':
#  	data_split = data_splitter.split_type(dataset, frac_train, frac_valid, frac_test, seed)
#  else: 
#    raise error # might want to expand this so we can account for edge cases like use of the 'generate_scaffolds' function

  # ensure k parameter is only being passed if all other parameters are set appropriately
  # this should catch many (but not all) inappropriate input combinations
    
#  if k is not None:
#  	if split_type != 'k_fold_split':
#  		raise error # k should only be passed if using k_fold_split
#  	elif frac_train is not None:
#  		raise error # cannot have both be true
#  	elif frac_valid is not None:
#  		raise error # connot have both be true
#  	elif frac_test is not None:
#  		raise error # cannot have both be true
#  	elif seed is not None:
#  		raise error # cannot have both be true
#  	else continue 
#  else: 
#    continue

  # this will catch an additional inappropriate input combination not caught by previous if statement  
#  if (((frac_valid is not None) or (frac_test is not None)) and split_type is not 'train_valid_test_split'):
#  	raise error # frac_valid and frac_test can only be passed if split_type is set to 'train_valid_test_split'
#  else: 
#    continue

#  return split_data


#def pipeline(datafile: str, id_cols: list of str, descriptor_cols: list of str, target_cols: list of str, **kwargs):
	"""
	Wrapping function for dataprep py files:
	- io.py
	- featurize.py
	- dataprep.py
	# note: may want to include analyze.py somehow

	Parameters
	----------
	datafrile: (str) datafile location to be converted to dataframe, then deepchem dataset.
	id_cols: (str or List[str]) columns to set as id column and check for repeats.
	descriptor_cols: (str or List[str]) columns with descriptor values.
	target_cols: (str or List[str]) columns with target values.
	**kwargs: keyword arguments for wrapped functions.
	# note: will likely need to be added to/updated as wrapped functions are written.

	Returns
	-------
	Dataset(s) preprocessed and ready for ml.
	"""

#	cols = [id_cols, descriptor_cols, target_cols,]
	
#	dataframe = load_file(datafile, cols)
#	load_XXX() # will need to see how this is written to know where it fits in
#	dataframe = clean_dataframe(dataframe, cols, duplicates = drop, nans = drop)
#	dataframe = featurize(dataframe, target_cols, id_col,)
#	dataframe = convert_to_categorical(dataframe, cols) # this 'cols' value should not be confused with the above 'cols' list
#	dataset = convert_to_dataset(dataframe, ) # additional arguments will have to be added to the wrapping function
#	ml_ready_dataset = preprocess(dataset, ) # additional arguments will have to be added to the wrapping function

#	return ml_ready_dataset
