from typing import Union, List

import deepchem as dc
import numpy as np
import pandas as pd
from sklearn import preprocessing
import rdkit.Chem


def convert_to_categorical(dataframe, cols=None):
    """
    Converts non-numerical categorical values to integers.  This function
    is most useful for ordinal variables.

    Parameters
    ----------
    - dataframe: featurized dataframe
    - cols: list of categorical columns to convert to integers

    Returns
    -------
    - modified dataframe with user selected columns with categorical string
      values converted to integer values
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
                    dataframe[col].shape) == 1:  # for 1D arrays
                # define label encoder
                encoder = preprocessing.LabelEncoder()
                # create new columns and preserve the original columns
                dataframe.loc[
                    :, col + '_encoded'
                ] = encoder.fit_transform(dataframe[col])
            else:
                # for 2D arrays (usually X, input features)
                # define ordinal encoder
                encoder = preprocessing.LabelEncoder()
                # create new columns and preserve the original columns
                dataframe.loc[
                    :, col + '_encoded'
                ] = encoder.fit_transform(dataframe[col])

    else:
        pass

    return dataframe


def binarize_targets(dataframe,
                     target_cols: Union[str, List[str]],
                     high_positive: bool = False,
                     percentile: float = 0.5,
                     value: Union[float, List[float]] = None):
    """Binarize target columns for classification.

    For targets of continuous variables, binarize based on a position in the
    distribution.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset with target columns.
    target_cols : str or list of str
        The column names to binarize.
    high_positive : bool
        If the end of the distribution higher than the chosen position should
        considered a posative target. eg. if True: posatives labeled for data
        > the postition
    percentile : float, default = 0.5 (median)
        The relative position in the distribution to consider for the two
        classes
    value : float, default None
        Specific value(s) to use as the cutoff for classes instead of percent.

    Example
    -------
    For target column of [ 0.0, 2.5, 5.0, 7.5, 10.0 ] with mean 5.0
    position = 0.75, high posative = True
    75% of the min max range is labeled as negative:
    labels = [ False, False, False, False, True ]
    """
    assert isinstance(dataframe, pd.core.frame.DataFrame),\
        'dataframe should be dataframe'
    if not isinstance(target_cols, list):
        target_cols = [target_cols]
    assert all([col in dataframe.columns for col in target_cols]),\
        'Not all columns are in the dataframe'
    assert all([dtype == float for dtype in dataframe[target_cols].dtypes]),\
        'Target columns should have float data types'
    dataframe_ = dataframe.copy()
    subset = dataframe_[target_cols]
    # the user wants to specify specific values
    if value is not None:
        if isinstance(value, float):
            pass
        else:
            value = np.array(value)
            assert len(value) == len(target_cols),\
                "Multiple values were specified for class cutoff but do not\
 match the number of target columns."
    else:
        value = subset.quantile(percentile).values
        
    # handle sparsity after values computed
    if np.sum(dataframe.isna()[target_cols].values) > 0:
        dataframe_ = handle_sparsity(dataframe_, y_col=target_cols)
    
    # now mask the targets
    dataframe_[target_cols] = subset > value
    # maybe switch
    if not high_positive:
        dataframe_[target_cols] = ~dataframe_[target_cols]
    return dataframe_


def canonicalize_smiles(smiles, raise_error=False):
    """Canonicalize a smiles string.

    Parameters
    ----------
    smiles : str

    rais_error : bool
        If canonicalizing fails whether to raise the error or simply return nan

    Returns
    -------
    csmiles : str
        Canonicalized smiles string.
    """
    try:
        assert isinstance(smiles, str),\
            f"smiles must be a string, not {type(smiles)}"
        mol = rdkit.Chem.MolFromSmiles(smiles)
        csmiles = rdkit.Chem.MolToSmiles(mol)
    except BaseException:
        if raise_error:
            raise
        else:
            csmiles = None
    return csmiles


def handle_sparsity(dataset, y_col=None, w_label='w'):
    """Prepares sparse data to be learned.

    Replace nans with 0.0 in the dataset targets so it can be input to a model,
    and scale the weight matrix with all nan values as 0.0 weight so that they
    do not introduce bias.

    Parameters
    ----------
    dataframe : dc.NumpyDataset or DataFrame
        The dataset with sparse targets.
    y_col : list of str
        The names of all columns containing the targets. (for df)
    w_label : str
        The string to add to the target names to create columns of weights in
        the dataframe.

    Returns
    -------
    dataset
    """
    if type(dataset) == dc.data.datasets.NumpyDataset:
        X, y, w, i = (dataset.X, dataset.y, dataset.w, dataset.ids)
        nans = np.isnan(y)
        # w may be only a vector instead of shape of data
        w = np.tile(w, y.shape[1]) 
        w *= ~nans
        y = np.nan_to_num(y)
        dataset_out = dc.data.NumpyDataset(X, y, w, i)
    elif type(dataset) == pd.core.frame.DataFrame:
        dataset_out = dataset.copy()
        if type(y_col) != list:
            y_col = [y_col]
        assert all([col in dataset.columns for col in y_col])
        w_names = [w_label + '_' + target for target in y_col]
        # compute weights based on presence of nan
        dataset_out[w_names] = np.float64(
            ~dataset_out[y_col].isnull().values
        )
        # It does not matter what value we replace the nans with as the weight is
        # 0, but it has to be numeric to not break the models
        dataset_out = dataset_out.fillna(0)
    return dataset_out


def convert_to_dataset(dataframe,
                       X_col: str = 'X',
                       y_col: str = 'y',
                       w_col: str = None,
                       w_label: str = None,
                       id_col: str = None):
    """
    Converts dataframe into a deepchem dataset object.

    Parameters
    ----------
    - dataframe: featurized dataframe
    - X_col: (str or List[str]) name(s) of the column(s) containing X.
    - y_col: (str or List[str]) name(s) of the column(s) containing y.
    - w_col: (str or List[str]) name(s) of the column(s) containing w.
    - w_label: str of the preceding label of target weight columns.
        ex. if the target is 'LD50' and the w_label is 'w' the datafame must
        contain a column of 'w_LD50'.
    - id_col: (str) name of the column containing the ids.

    Returns
    -------
    - dataset: deepchem dataset object
    """
    # define x
    if isinstance(X_col, str):
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
    if isinstance(y_col, str):
        y_col = [y_col]
    y = dataframe[y_col].values
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # define weight
    if w_col is not None:
        w = np.vstack(dataframe[w_col].values)
    elif w_label is not None:
        w_col = [w_label + '_' + target for target in y_col]
        w = np.vstack(dataframe[w_col].values)
    else:
        w = None

    # define ids
    if id_col is not None:
        ids = dataframe[id_col].values
    else:
        ids = np.array(dataframe.index)

    # create deepchem dataset object
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    return dataset


def data_transformation(dataset,
                        transformations: list = ['NormalizationTransformer'],
                        to_transform: Union[str, List[str]] = None,
                        **kwargs):
    """
    Transforms and splits deepchem dataset

    Parameters
    ----------
    - dataset: deepchem dataset to be transformed
    - transformations:
        (List[str]) list of transformation methods to pass dataset through
    - to_transform:
        (str or list[str]) list of elements to transform, and can include
        'X', 'y', or 'w'. If multiple are specified, must be paried to
        transformations
    - **kwargs: keyword arguments to be passed to the selected transformer

    Returns
    -------
    - transformed dataset
    - list of transformer objects
    """
    if type(transformations) == str:
        transformations = [transformations]
    elif type(transformations) == list:
        if len(transformations) == 0:
            return dataset, []
    if to_transform is not None:
        if type(to_transform) == str:
            to_transform = [to_transform]*len(transformations)
        elif type(to_transform) == list:
            assert len(to_transform) == len(transformations)

    # make a list to store transformer object, which can later be used to
    # untransform data
    transformer_list = []

    # feed dataset into list of transformers sequentially, returning a single
    # transformed dataset
    for i, transformation in enumerate(transformations):
        if to_transform is not None:
            to_transform_ = to_transform[i] 

            if(all(elem in to_transform_ for elem in ['X', 'y', 'w'])):
                transformer = getattr(
                    dc.trans,
                    transformation)(
                    transform_X=True,
                    transform_y=True,
                    transform_w=True,
                    dataset=dataset,
                    **kwargs)
            elif(all(elem in to_transform_ for elem in ['X', 'y'])):
                transformer = getattr(
                    dc.trans,
                    transformation)(
                    transform_X=True,
                    transform_y=True,
                    dataset=dataset,
                    **kwargs)
            elif 'X' in to_transform_:
                transformer = getattr(
                    dc.trans, transformation)(
                    transform_X=True, dataset=dataset, **kwargs)
            elif 'y' in to_transform_:
                transformer = getattr(
                    dc.trans, transformation)(
                    transform_y=True, dataset=dataset, **kwargs)
            else:
                raise ValueError(
                    'to_transform was specified but did not contain\
 exclusively X, y, and w.'
                )
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
    - split_type:
        (str) type of split
        (k_fold_split/train_test_split/train_valid_test_split)
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
    # the 'split' option is excluded since it seems to do the same thing as
    # 'train_valid_test_split' but returns non-dataset objects
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
                'split_type may only be set as generate_scaffolds if splitter\
 set as ScaffoldSplitter'
            )
    else:
        # should change this to raise an error of some kind
        print('split_type string is not a recognized split')

    return data_split


# def dataset_prep(dataframe,
#                  transformations: list = None,
#                  to_transform: list = None,
#                  input_features=None,
#                  label=None,
#                  weights=None,
#                  id_col=None,
#                  splitter=None,
#                  splitter_type=None,
#                  **kwargs):
#     """
#     Wrapping functions for convert_to_dataset, data_transformation, and data_splitting

#     Parameters
#     ----------
#     - dataframe: dataframe to be converted to dataset object
#     - transformations: (List[str]) list of transformation methods to pass dataset through
#     - to_transform: (list[str]) list of elements to transform, and can include 'X', 'y', or 'w'
#     - input_features: (str or List[str]) name(s) of the column(s) containing the X array.
#     - label: (str or List[str]) name(s) of the column(s) containing the y array.
#     - weights: (str or List[str]) name(s) of the column(s) containing the w array.
#     - id_col: (str) name of the column containing the ids.
#     - splitter: (str) class of deepchem split method
#     - split_type: (str) type of split (k_fold_split/train_test_split/train_valid_test_split)

#     Returns
#     -------
#     - split_data: tuple or list containing dataset splits
#     - transformer_list: list of transformer objects used
#     """

#     # convert dataframe to dataset object
#     dataset = convert_to_dataset(
#         dataframe=dataframe,
#         X_col=input_features,
#         y_col=label,
#         w_col=weights,
#         id_col=id_col)

#     # transform data
#     if transformations is not None:
#         transformed_dataset, transformer_list = data_transformation(
#             dataset=dataset, transformations=transformations, to_transform=to_transform, **kwargs)
#     else:
#         transformed_dataset = dataset, transformer_list = []
#         print('no transformations performed')

#     # split data
#     split_data = data_splitting(
#         dataset=transformed_dataset,
#         splitter=splitter,
#         split_type=splitter_type,
#         **kwargs)

#     return split_data, transformer_list
