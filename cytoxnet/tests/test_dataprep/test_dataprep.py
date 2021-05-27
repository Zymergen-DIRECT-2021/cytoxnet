"""
Tests for dataprep.py
"""


from cytoxnet.dataprep import dataprep
import math
import deepchem as dc
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """
    Import sample dataframes for use in test_dataprep.py functions
    """

    df = pd.DataFrame({'smiles': ['C', 'O'],
                       'target': [1.0, 2.0],
                       'target2': [3.0, 4.0],
                       'target3': [None, 1.0],
                       'cf': [np.array([1, 2]),
                              np.array([3, 4])],
                       'w': [.5, .75]})
    return df


def test_binarize_targets(sample_data):
    """Converting column to binary target."""
    df = sample_data

    # first try specified percentile
    subject = dataprep.binarize_targets(
        df,
        target_cols='target',
        high_positive=True,
        percentile=0.5
    )
    assert np.array_equal(subject['target'].values,
                          np.array([False, True]))
    # flip the posative
    subject = dataprep.binarize_targets(
        df,
        target_cols='target',
        high_positive=False,
        percentile=0.5
    )
    assert np.array_equal(subject['target'].values,
                          np.array([True, False]))

    # specify a value and multiple columns
    subject = dataprep.binarize_targets(
        df,
        target_cols=['target', 'target2'],
        value=1.5,
        high_positive=True
    )
    assert np.array_equal(subject[['target', 'target2']].values,
                          np.array([[False, True],
                                    [True, True]]))
    subject = dataprep.binarize_targets(
        df,
        target_cols=['target', 'target2'],
        value=[0.0, 3.5],
        high_positive=True
    )
    assert np.array_equal(subject[['target', 'target2']].values,
                          np.array([[True, False],
                                    [True, True]]))
    return


def test_canonicalize_smiles():
    """Valid smiles should be canonicalized."""
    assert dataprep.canonicalize_smiles('C') == 'C',\
        "Canonicalization failed, was already canonical"
    assert dataprep.canonicalize_smiles('OC') == 'CO',\
        "Canonicalization failed"
    return


def test_handle_sparsity(sample_data):
    """Sparse datapoints should be weighted zero."""
    df = sample_data
    subject = dataprep.handle_sparsity(
        df,
        y_col=['target', 'target3'],
        w_label='string'
    )
    assert all(
        [wc in subject.columns for wc in ['string_target', 'string_target3']]
    ), 'Weight columns were not added.'
    assert np.array_equal(subject['string_target'], np.array([1.0, 1.0])),\
        "None sparse target has incorrect weight column."
    assert np.array_equal(subject['string_target3'], np.array([0.0, 1.0])),\
        "sparse target was not unweighted."
    return


def test_convert_to_dataset(sample_data):
    """
    Test convert_to_dataset function
    """

    # create dataframe
    df = sample_data

    # convert dataframe to dataset
    dataset = dataprep.convert_to_dataset(
        dataframe=df,
        X_col='cf',
        y_col='target')

    assert isinstance(dataset, dc.data.datasets.NumpyDataset), 'Dataset is not\
        deepchem NumpyDataset object'
    assert dataset.X.shape == (2, 2)
    assert dataset.y.shape == (2, 1)
    assert np.array_equal(dataset.w, np.array([[1.0], [1.0]]))

    # try specifyin a weight column
    dataset = dataprep.convert_to_dataset(
        dataframe=df,
        X_col='cf',
        y_col='target',
        w_col='w'
    )
    assert np.array_equal(dataset.w, np.array([[.5], [.75]]))

    # and a wieght prefix
    df = dataprep.handle_sparsity(
        df,
        y_col=['target3'],
        w_label='w'
    )
    dataset = dataprep.convert_to_dataset(
        dataframe=df,
        X_col='cf',
        y_col='target3',
        w_label='w'
    )
    assert np.array_equal(dataset.w, np.array([[0.], [1.0]]))
    return


def test_data_transformation(sample_data):
    """
    Test data_transformation function
    """

    # create dataframe
    df = sample_data

    # convert dataframe to dataset
    dataset = dataprep.convert_to_dataset(
        dataframe=df,
        X_col='cf',
        y_col='target',
        w_col=None,
        id_col=None)

    # transform feature data using Normalization Transformer first, followed
    # by Min-Max Transformer
    transformed_data, transformer_list = dataprep.data_transformation(
        dataset=dataset, transformations=[
            'NormalizationTransformer', 'MinMaxTransformer'
        ], to_transform=['X'])

    assert all(
        [isinstance(
            transformer, dc.trans.transformers.Transformer
        ) for transformer in transformer_list]), 'Transformers not returned.'

    return


def test_data_splitting(sample_data):
    """
    Test data_splitting function
    """

    # create dataframe
    df = sample_data

    # convert dataframe to dataset
    dataset = dataprep.convert_to_dataset(
        dataframe=df,
        X_col='cf',
        y_col='target',
        w_col=None,
        id_col=None)

    # split data using k-fold split
    split = dataprep.data_splitting(
        dataset=dataset,
        splitter='RandomSplitter',
        split_type='k',
        k=2)

    # split data using train-test split
    train, test = dataprep.data_splitting(
        dataset=dataset,
        splitter='RandomSplitter',
        split_type='train_test_split',
        frac_train=0.5
    )

    assert np.shape(split) == (2, 2), 'k_fold_split is\
       wrong shape'
    assert math.isclose(0.5, len(
        train) / (len(test) + len(train)), rel_tol=0.05), 'Train-Test split\
 has incorrect proportions'
    return
