"""
Tests for dataprep.py
"""

import os

from cytoxnet.dataprep import dataprep
from cytoxnet.dataprep import io
from cytoxnet.dataprep import featurize
import math


def test_convert_to_dataset():
    """
    Test convert_to_dataset function
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, '..', 'data', 'chembl_example.csv')

    df = io.load_data(filename,
                      col_id='Molecule ChEMBL ID')
    df_1 = featurize.molstr_to_Mol(df, strcolumnID='Smiles')
    df_2 = featurize.add_features(df_1)

    dataset, csv = dataprep.convert_to_dataset(
        dataframe=df_2, X_col='CircularFingerprint', y_col='Standard Value', w_col=None, id_col=None, return_csv=True)

    assert type(csv) is str, 'CSV not returned when specified'
    assert isinstance(dataset, dc.data.datasets.NumpyDataset), 'Dataset is not\
        deepchem NumpyDataset object'
    assert dataset.X.shape[1] > 0, 'Dataset is incorrect shape'
    assert dataset.X.shape[0] > 0, 'Dataset is incorrect shape'
    return


def test_data_transformation():
   """
   Test data_transformation function
   """

   dir_path = os.path.dirname(os.path.realpath(__file__))
   filename = os.path.join(dir_path, '..', 'data', 'chembl_example.csv')

   df = io.load_data(filename,
                     col_id='Molecule ChEMBL ID')
   df_1 = featurize.molstr_to_Mol(df, strcolumnID='Smiles')
   df_2 = featurize.add_features(df_1)

   dataset, csv = dataprep.convert_to_dataset(
       dataframe=df_2, X_col='CircularFingerprint', y_col='Standard Value', w_col=None, id_col=None, return_csv=False)

   transformed_data, transformer_list = dataprep.data_transformation(
       dataset=dataset, transformations=[
           'NormalizationTransformer', 'MinMaxTransformer'], to_transform=['X'])

   assert isinstance(
       transformer_list[0], dc.trans.transformers.NormalizationTransformer), 'Transformer list\
       ordered incorrectly'
   return


def test_data_splitting():
   """
   Test data_splitting function
   """

   dir_path = os.path.dirname(os.path.realpath(__file__))
   filename = os.path.join(dir_path, '..', 'data', 'chembl_example.csv')

   df = io.load_data(filename,
                     col_id='Molecule ChEMBL ID')
   df_1 = featurize.molstr_to_Mol(df, strcolumnID='Smiles')
   df_2 = featurize.add_features(df_1)

   dataset, csv = dataprep.convert_to_dataset(
       dataframe=df_2, X_col='CircularFingerprint', y_col='Standard Value', w_col=None, id_col=None, return_csv=False)

   transformed_data, transformer_list = dataprep.data_transformation(
       dataset=dataset, transformations=['NormalizationTransformer'], to_transform=['X'])

   split = dataprep.data_splitting(
       dataset=transformed_data,
       splitter='RandomSplitter',
       split_type='k',
       k=7)

   train, test = dataprep.data_splitting(
       dataset=transformed_data, splitter='RandomSplitter', split_type='train_test_split', frac_train=0.4)
   a = train.X.shape[0] / test.X.shape[0]
   b = 4

   assert np.shape(split) == (7, 2), 'k_fold_split is\
       wrong shape'
   assert math.isclose(a, b, rel_tol=0.05), 'Train-Test split has\
       incorrect proportions'
   return
