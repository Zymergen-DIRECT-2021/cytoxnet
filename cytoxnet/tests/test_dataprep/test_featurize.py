"""
Tests for featurize.py
"""

from cytoxnet.dataprep import io
from cytoxnet.dataprep import featurize
import rdkit

def test_molstr_to_Mol():
    """
    Test molstr_to_Mol function for adding Mol objects 
    corresponding to smiles or inchi strings in a dataframe
    """
    df = io.load_data('../data/chembl_example.csv',
                  col_id='Molecule ChEMBL ID')
    df_1 = featurize.molstr_to_Mol(df, strcolumnID='Smiles')
    assert 'Mol' in df_1.columns, 'No Mol column created'
    assert isinstance(df_1['Mol'][1], rdkit.Chem.Mol), 'Mols are not\
        rdkit Mol objects'
    return


def test_add_features():
    df = io.load_data('../data/chembl_example.csv',
                  col_id='Molecule ChEMBL ID')
    df_1 = featurize.molstr_to_Mol(df, strcolumnID='Smiles')
    df_2 = featurize.add_features(df_1)
    
    assert 'CircularFingerprint' in df_2.columns, 'Correct default\
        feature column not created'
    assert len(df_2['CircularFingerprint'][1][0]) == 2048, 'Wrong default\
        fingerprint length'
    return
