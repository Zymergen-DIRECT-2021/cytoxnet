"""
Tests for io.py.
"""

from cytoxnet.dataprep import io


def test_load_data():
    """
    Test the load_data function.
    """
    df = io.load_data('../data/chembl_example.csv')
    assert len(df.index) == 99, 'Incorrect number of lines in resulting DF'
    assert df['Molecule ChEMBL ID'][0] == 'CHEMBL3290077', 'Incorrect data\
        in loaded dataframe'
    assert len(df.columns) == 46, 'Incorrect number of columns in DF'

    df_2 = io.load_data('../data/chembl_example.csv',
                        col_id='Molecule ChEMBL ID')
    assert len(df_2.index) == 24, 'Incorrect duplicate removal'
    assert df_2['Molecule ChEMBL ID'][23] == 'CHEMBL617', 'Incorrect data\
        in loaded dataframe after duplicates removed'
    assert len(df_2.columns) == 46, 'Incorrect number of columns in DF\
        after fduplicate rows removed'

    test1 = False
    try:
        io.load_data('bad_filename.csv')
    except Exception as e:
        assert isinstance(e, AssertionError), "Wrong type of error."
        test1 = True
        assert test1, "Test failed!\
        The load_data function is not\
        responsive to a wrong file name"

    return
