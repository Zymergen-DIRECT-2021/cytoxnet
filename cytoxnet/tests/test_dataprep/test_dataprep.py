"""
Tests for dataprep.py
"""

import os

from cytoxnet.dataprep import dataprep
from cytoxnet.dataprep import io
from cytoxnet.dataprep import featurize
import math
import deepchem as dc
import numpy as np
import pytest


@pytest.fixture
def sample_data():
    """
    Import sample dataframes for use in test_dataprep.py functions
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, '..', 'data', 'sample_df1')
    df = pd.read_csv(filename)
    return df


@pytest.fixture
def test_convert_to_dataset(sample_data):
    """
    Test convert_to_dataset function
    """

    # create dataframe
    df = sample_data

    # convert dataframe to dataset
    dataset = dataprep.convert_to_dataset(
        dataframe=df,
        X_col='CircularFingerprint',
        y_col='Standard Value',
        w_col=None,
        id_col=None)

    assert isinstance(dataset, dc.data.datasets.NumpyDataset), 'Dataset is not\
        deepchem NumpyDataset object'
    assert dataset.X.shape[1] > 0, 'Dataset is incorrect shape'
    assert dataset.X.shape[0] > 0, 'Dataset is incorrect shape'
    return


@pytest.fixture
def test_data_transformation(sample_data):
    """
    Test data_transformation function
    """

    # create dataframe
    df = sample_data

    # convert dataframe to dataset
    dataset = dataprep.convert_to_dataset(
        dataframe=df,
        X_col='CircularFingerprint',
        y_col='Standard Value',
        w_col=None,
        id_col=None)

    # transform feature data using Normalization Transformer first, followed
    # by Min-Max Transformer
    transformed_data, transformer_list = dataprep.data_transformation(
        dataset=dataset, transformations=[
            'NormalizationTransformer', 'MinMaxTransformer'], to_transform=['X'])

    assert isinstance(
        transformer_list[0], dc.trans.transformers.NormalizationTransformer), 'Transformer list\
       ordered incorrectly'
    return


@pytest.fixture
def test_data_splitting(sample_data):
    """
    Test data_splitting function
    """

    # create dataframe
    df = sample_data

    # convert dataframe to dataset
    dataset = dataprep.convert_to_dataset(
        dataframe=df_2,
        X_col='CircularFingerprint',
        y_col='Standard Value',
        w_col=None,
        id_col=None)

    # transform feature data using Normalization Transformer
    transformed_data, transformer_list = dataprep.data_transformation(
        dataset=dataset, transformations=['NormalizationTransformer'], to_transform=['X'])

    # split data using k-fold split
    split = dataprep.data_splitting(
        dataset=transformed_data,
        splitter='RandomSplitter',
        split_type='k',
        k=7)

    # split data using train-test split
    train, test = dataprep.data_splitting(
        dataset=transformed_data, splitter='RandomSplitter', split_type='train_test_split', frac_train=0.4)
    a = train.X.shape[0] / test.X.shape[0]
    b = 4

    assert np.shape(split) == (7, 2), 'k_fold_split is\
       wrong shape'
    assert math.isclose(a, b, rel_tol=0.05), 'Train-Test split has\
       incorrect proportions'
    return
