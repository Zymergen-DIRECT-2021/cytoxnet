# import pandas as pd
# import deepchem
# import rdkit


def molstr_to_Mol(dataframe, strcolumnID='InChI String'):
    """
    Converts DataFrame column containing the string representations of
    molecules and add a corresponding column of Mol objects.

    Parameters
    ----------
    dataframe: DataFrame containing a column with string representations
    of molecules
    strcolumnID: label for the column containg the string representations for
    converstion, default='InChI String'- can be changed for the user's
    standard column name

    Returns
    -------
    DataFrame with additional column containing Mol objects for each molecule,
    column label is 'Mol'

    """

    # get the column containting the string representation
    # check whether column title contains "inche" or "smiles"
    # if 'inchi' in strcolumnID.lower():
    #     convert InChI to Mol
    #    for inchi in column:
    #        mol_obj = rdkit.Chem.inchi.MolFromInchi(inchi)
    #        append to array first here probably
    # elif 'smiles' in strcpolumnID.lower():
    #    convert SMILES to Mol
    #    for smiles in column:
    #        mol_obj = MolFromSmiles(smiles)
    #        probably append to an array
    # else:
    #    print('Column ID does not contain inchi or smiles')

    # add the mol objects to the dataframe
    # dataframe['Mol'] = mol_obj_list

    return dataframe


def featurize(dataframe, MolcolumnID='Mol', method='CircularFingerprint'):
    """
    Featurizes a set of Mol objects using the desired feturization method.

    note: may want to change setup of parameters here
    note: may be better suited as a class
    note: might not need separate featurizer for graph featurization-
    could include here I think

    list of possible featurizers to wrap up here:
    https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html

    Parameters
    ----------
    dataframe: a DataFrame containing a column with Mol objects-
    may want to play around with how we want to input the set of Mols
    for featurization
    columnID: label of the column containing the Mol objects, default based
    on function for adding a Mol column

    Returns
    -------
    DataFrame containing a column of the featurized representation of the Mol
    object with the featurization method as the column ID

    """

    # get the set of mol objects by extracting the specified column
    # or iterating over it

    # Check that set contains Mol objects
    # assert isinstance(object_in_set, rdkit.Chem.rdchem.Mol)

    # featurizer = getattr(deepchem.feat, method)

    # molecules format optins-  rdkit.Chem.rdchem.Mol /
    # SMILES string / iterable
    # either a loop or pass an iterable set of Mol
    # iterable = convert column of Mol to array
    # features = featurizer.featurize(iterable)
    # add the featurized representation into the passed dataframe
    # dataframe[method] = features

    return dataframe


def get_descriptors(dataframe, descriptor_type):
    """
    Extracts molecular descriptors and adds them to a new column
    Something to look into in the future
    What descriptors might we want? how can we get them?
    will need a featurizing function that can take any descriptors we
    extract here
    """
    # dataframe[descriptor_type] = descriptorx_list
    return dataframe
