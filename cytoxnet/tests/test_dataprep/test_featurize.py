"""
Tests for featurize.py
"""

import os
import tempfile

import unittest.mock as mock

import numpy as np

from cytoxnet.dataprep import io
from cytoxnet.dataprep import featurize
import rdkit


def test_molstr_to_Mol():
    """
    Test molstr_to_Mol function for adding Mol objects
    corresponding to smiles or inchi strings in a dataframe
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, '..', 'data', 'minimum_data_example.csv')

    df = io.load_data(filename,
                      id_cols='smiles',
                      nans='drop')
    df_1 = featurize.molstr_to_Mol(df, id_col='smiles')
    assert 'Mol' in df_1.columns, 'No Mol column created'
    assert isinstance(df_1['Mol'][1], rdkit.Chem.Mol), 'Mols are not\
        rdkit Mol objects'
    return


def test_from_np_array():
    """Converting saved strings to arrays.
    """
    string = "[ 1.0 2.0 3.0 ]"
    subject = featurize.from_np_array(string)
    assert np.array_equal(subject, np.array([1, 2, 3]))
    bad_string = "abcd"
    subject = featurize.from_np_array(bad_string)
    assert subject is None
    return


def test_add_features():
    """Creating features from smiles string."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, '..', 'data', 'minimum_data_example.csv')

    df = io.load_data(filename,
                      id_cols='smiles',
                      nans='drop')

    # start with raw featurization - no codex
    df = featurize.add_features(df, id_col='smiles')
    assert 'CircularFingerprint' in df.columns, 'Correct default\
        feature column not created'
    assert len(df['CircularFingerprint'][1]) == 2048, 'Wrong default\
        fingerprint length'

    # try to add a codex
    with tempfile.TemporaryDirectory() as tempdir:
        io.create_compound_codex(
            db_path=tempdir + '/database',
            id_col='smiles',
            featurizers=['RDKitDescriptors']
        )
        io.add_datasets([df],
                        ['mydata'],
                        id_col='smiles',
                        db_path=tempdir + '/database')
        with mock.patch(
            'cytoxnet.dataprep.featurize.dc.feat.RDKitDescriptors'
        ) as mocked_rdkdesc:
            featurize.add_features(df,
                                   method='RDKitDescriptors',
                                   codex=tempdir + '/database/compounds.csv')

            args = mocked_rdkdesc().featurize.call_args
            assert len(args[0][0]) == 0,\
                "Should have loaded features and not called the featurizer"
    return
