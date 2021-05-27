"""Tests for io.py.
"""

import os

import pytest
import tempfile
import unittest.mock as mock

import numpy as np
import pandas as pd

import cytoxnet.dataprep.io
import cytoxnet.data




def test_load_data():
    """Test the load_data function.
    
    Should be able to find files, and package data. Also dropping nans
    if asked.
    """
    # test with specified path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, '..', 'data', 'minimum_data_example.csv')
    
    # specify the columns to take, don't drop nans
    df = cytoxnet.dataprep.io.load_data(filename, cols=['smiles'])
    assert len(df.columns) == 1,\
        'Correct no. of columns was not loaded'
    assert len(df) == 3,\
        'Wrong number of rows loaded'
    # and to drop nans
    df = cytoxnet.dataprep.io.load_data(filename, id_cols=['smiles'], nans='drop')
    assert len(df) == 2,\
        'Nans in the specified id columns were not dropped'
    
    # test with package data
    datapath = cytoxnet.data.__path__._path[0]
    with mock.patch('os.listdir',
                    return_value = ['myfile1.csv', 'myfile2.csv']
                   ) as mocked_os:
        with mock.patch('cytoxnet.dataprep.io.pd') as mocked_pandas:
            cytoxnet.dataprep.io.load_data('myfile2')
            mocked_pandas.read_csv.assert_called_with(
                datapath+'/myfile2.csv',
                index_col=0
            )
    
    # bad file name
    with pytest.raises(FileNotFoundError):
        cytoxnet.dataprep.io.load_data('astring')
        
    return

@mock.patch('cytoxnet.dataprep.io.pd')
@mock.patch('cytoxnet.dataprep.io.os')
def test_create_compound_codex(mocked_os, mocked_pandas):
    """Initialization of compounds codex.
    
    Should create empty file at specified location containing the requested
    id_col and featurizers.
    """
    # no features
    mocked_os.path.exists.return_value = False
    cytoxnet.dataprep.io.create_compound_codex(db_path='./database',
                                               id_col='smiles')
    mocked_os.makedirs.assert_called_with('./database')
    mocked_pandas.DataFrame.assert_called_with(
        columns=['smiles']
    )
    mocked_pandas.DataFrame().to_csv.assert_called_with(
        './database/compounds.csv'
    )
    mocked_os.reset_mock()
    #features
    mocked_os.path.exists.return_value = True
    cytoxnet.dataprep.io.create_compound_codex(
        db_path='./database',
        id_col='smiles',
        featurizers=['CircularFingerprint']
    )
    assert not mocked_os.makedirs.called,\
        "Should not have made a dir"
    mocked_pandas.DataFrame.assert_called_with(
        columns=['smiles', 'CircularFingerprint']
    )
    return

def test_add_datasets(tmpdir):
    """These should add non duplicate smiles to the compounds list.
    
    Added datasets should also be assigned keys and added to the dataset.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(dir_path, '..', 'data', 'minimum_data_example.csv')
        df = cytoxnet.dataprep.io.load_data(filename, id_cols=['smiles'],
                                            nans='drop')

        # test new data computing old features
        cytoxnet.dataprep.io.create_compound_codex(
            db_path=tempdir+'/database',
            id_col='smiles',
            featurizers=['CircularFingerprint']
        )
        cytoxnet.dataprep.io.add_datasets([df],
                                          ['mydata'],
                                          id_col='smiles',
                                          db_path=tempdir+'/database')
        subject = pd.read_csv(tempdir+'/database/compounds.csv', index_col=0)
        assert len(subject) == 2,\
            'Not all smiles were added'
        assert not subject['CircularFingerprint'].isnull().any(),\
            'Did not compute features'
        subject = pd.read_csv(tempdir+'/database/mydata.csv', index_col=0)
        assert np.array_equal(subject['foreign_key'].values, [0,1]),\
            'foreign keys not assigned properly'

        # test new data and new feature
        filename = os.path.join(dir_path, '..', 'data', 'minimum_data_example2.csv')
        df = cytoxnet.dataprep.io.load_data(filename, id_cols=['smiles'],
                                            nans='drop')
        cytoxnet.dataprep.io.add_datasets([df],
                                          ['mydata2'],
                                          id_col='smiles',
                                          db_path=tempdir+'/database',
                                          new_featurizers=['RDKitDescriptors'])
        # this addition has a duplicate smiles, so should only add 1 to compounds
        subject = pd.read_csv(tempdir+'/database/compounds.csv', index_col=0)
        assert len(subject) == 3,\
            'Smiles not properly added - should have only added 1'
        assert not subject['RDKitDescriptors'].isnull().any(),\
            'Did not compute features'
        subject = pd.read_csv(tempdir+'/database/mydata2.csv', index_col=0)
        assert np.array_equal(subject['foreign_key'].values, [2,1]),\
            'foreign keys not assigned properly'
        
        # add package data
        with mock.patch(
            'cytoxnet.dataprep.io.load_data',
            return_value=pd.DataFrame({'smiles':['C', 'O']})
        ) as mocked_load_data:
            cytoxnet.dataprep.io.add_datasets(['lunghini_algea_EC50'],
                                              ['mydata3'],
                                              id_col='smiles',
                                              db_path=tempdir+'/database')
            assert mocked_load_data.called,\
                'Load data was not called for the string'

    return
