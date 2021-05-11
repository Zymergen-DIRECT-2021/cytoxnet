import deepchem as dc
import numpy as np
import pandas as pd
from sklearn import preprocessing

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
        if isinstance(
                cols, str):  # allow single columns to be input as strings
            cols = [cols]
        else:
            pass

        for col in cols:
            if len(
                    dataframe[col].shape) == 1:  # for 1D arrays (usally y, target variables)
                # define label encoder
                encoder = preprocessing.LabelEncoder()
                # create new columns and preserve the original columns
                dataframe.loc[:, col +
                              '_encoded'] = encoder.fit_transform(dataframe[col])
            else:  
                # for 2D arrays (usually X, input features)
                # define ordinal encoder
                encoder = preprocessing.LabelEncoder()
                # create new columns and preserve the original columns
                dataframe.loc[:, col +
                              '_encoded'] = encoder.fit_transform(dataframe[col])

    else:
        pass

    return dataframe

def handle_sparsity(dataframe,
                    y_col: Union[str, List[str]],
                    weight_col: str = 'w'):
    """Prepares sparse data to be learned.
    
    Replace nans with 0.0 in the dataset so that it can be input to a model,
    and create a weight matrix with all nan values as 0.0 weight so that they
    do not introduce bias.
    """

def convert_to_dataset(dataframe,
                       X_col: str = 'X',
                       y_col: str = 'y',
                       w_col: str = None,
                       id_col: str = None):
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
    - dataset: deepchem dataset object
    """
    # define x
    if type(X_col) == str:
        X_col = [X_col]
    X_list = []
    for col in X_col:
        X_ = dataframe[col].values
        X_ = np.vstack(X_)
        X_list.append(X_)
    X = np.hstack(X_list)
    # need to check for object features
    if not np.issubdtype(X.dtype, np.number):
        X = X.reshape(-1)

    # define y
    y = dataframe[y_col].values

    # define weight
    if w_col is not None:
        w = np.vstack(dataframe[w_col].values)
    else:
        w = None

    # define ids
    if id_col is not None:
        ids = dataframe[id_col].values
    else:
        ids = None

    # create deepchem dataset object
    dataset = dc.data.NumpyDataset(X, y, w, ids)

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
    - transformed dataset
    - list of transformer objects
    """

    # make a list to store transformer object, which can later be used to
    # untransform data
    transformer_list = []

    # feed dataset into list of transformers sequentially, returning a single
    # transformed dataset
    for transformation in transformations:
        if to_transform is not None:

            if(all(elem in to_transform for elem in ['X', 'y', 'w'])):
                transformer = getattr(
                    dc.trans,
                    transformation)(
                    transform_X=True,
                    transform_y=True,
                    transform_w=True,
                    dataset=dataset,
                    **kwargs)
            elif(all(elem in to_transform for elem in ['X', 'y'])):
                transformer = getattr(
                    dc.trans,
                    transformation)(
                    transform_X=True,
                    transform_y=True,
                    dataset=dataset,
                    **kwargs)
            elif 'X' in to_transform:
                transformer = getattr(
                    dc.trans, transformation)(
                    transform_X=True, dataset=dataset, **kwargs)
            elif 'y' in to_transform:
                transformer = getattr(
                    dc.trans, transformation)(
                    transform_y=True, dataset=dataset, **kwargs)
            else:
                transformer = getattr(
                    dc.trans, transformation)(
                    dataset=dataset, **kwargs)
        else:
            transformer = getattr(
                dc.trans,
                transformation)(
                dataset=dataset,
                **kwargs)

        transformer_list.append(transformer)

        dataset = transformer.transform(dataset)

    return dataset, transformer_list


def data_splitting(dataset,
                   splitter: str = 'RandomSplitter',
                   split_type: str = 'train_valid_test_split',
                   **kwargs):
    """
    Transforms and splits deepchem dataset

    Parameters
    ----------
    - dataset: deepchem dataset to be split
    - splitter: (str) class of deepchem split method
    - split_type: (str) type of split (k_fold_split/train_test_split/train_valid_test_split)
    - **kwargs: keyword arguments to be passed to the selected splitter

    Returns
    -------
    Split dataset
    """

    # split data
    data_splitter = getattr(dc.splits, splitter)

    try:
        split = data_splitter(**kwargs)
    except TypeError:
        split = data_splitter()

    # this only allows the following three split_types to be used
    # the 'split' option is excluded since it seems to do the same thing as 'train_valid_test_split' but returns non-dataset objects
    # might want to write code to reset certain defaults depending on which
    # split_type is chosen (i.e. from None to 0.8)

    if split_type == 'k_fold_split' or split_type == 'k':
        data_split = split.k_fold_split(dataset=dataset, **kwargs)
    elif split_type == 'train_test_split' or split_type == 'tt':
        data_split = split.train_test_split(dataset=dataset, **kwargs)
    elif split_type == 'train_valid_test_split' or split_type == 'tvt':
        data_split = split.train_valid_test_split(dataset=dataset, **kwargs)
    elif split_type == 'generate_scaffolds':
        if hasattr(split, 'generate_scaffolds'):
            # Unsure about functionality, code may need to be added
            data_split = split.generate_scaffolds(dataset=dataset, **kwargs)
        else:
            raise AttributeError(
                'split_type may only be set as generate_scaffolds if splitter set as ScaffoldSplitter')
    else:
        # should change this to raise an error of some kind
        print('split_type string is not a recognized split')

    return data_split


def dataset_prep(dataframe,
                 transformations: list = None,
                 to_transform: list = None,
                 input_features=None,
                 label=None,
                 weights=None,
                 id_col=None,
                 splitter=None,
                 splitter_type=None,
                 **kwargs):
    """
    Wrapping functions for convert_to_dataset, data_transformation, and data_splitting

    Parameters
    ----------
    - dataframe: dataframe to be converted to dataset object
    - transformations: (List[str]) list of transformation methods to pass dataset through
    - to_transform: (list[str]) list of elements to transform, and can include 'X', 'y', or 'w'
    - input_features: (str or List[str]) name(s) of the column(s) containing the X array.
    - label: (str or List[str]) name(s) of the column(s) containing the y array.
    - weights: (str or List[str]) name(s) of the column(s) containing the w array.
    - id_col: (str) name of the column containing the ids.
    - splitter: (str) class of deepchem split method
    - split_type: (str) type of split (k_fold_split/train_test_split/train_valid_test_split)

    Returns
    -------
    - split_data: tuple or list containing dataset splits
    - transformer_list: list of transformer objects used
    """

    # convert dataframe to dataset object
    dataset = convert_to_dataset(
        dataframe=dataframe,
        X_col=input_features,
        y_col=label,
        w_col=weights,
        id_col=id_col)

    # transform data
    if transformations is not None:
        transformed_dataset, transformer_list = data_transformation(
            dataset=dataset, transformations=transformations, to_transform=to_transform, **kwargs)
    else:
        transformed_dataset = dataset, transformer_list = []
        print('no transformations performed')

    # split data
    split_data = data_splitting(
        dataset=transformed_dataset,
        splitter=splitter,
        split_type=splitter_type,
        **kwargs)

    return split_data, transformer_list
