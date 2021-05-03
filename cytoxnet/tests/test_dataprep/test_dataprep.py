"""
Tests for dataprep.py
"""

import os

from cytoxnet.dataprep import dataprep
import math


def test_convert_to_dataset():
    """
    Test convert_to_dataset function
    """

    # create example dataframe (values extracted randomly from existing ChEMBL query)
    example_dict = {'index': [4829, 5079, 731, 749, 3488], 'Molecule ChEMBL ID': ['CHEMBL1276627', 'CHEMBL4457967', 'CHEMBL2424894', 'CHEMBL2424889', 'CHEMBL3086842'], 'Molecule Name': [nan, nan, nan, nan, nan], 'Molecule Max Phase': [0, 0, 0, 0, 0], 'Molecular Weight': ['406.53', '880.11', '450.51', '497.58', '232.24'], '#RO5 Violations': ['0', '2', '0', '0', '0'], 'AlogP': ['4.12', '3.74', '2.66', '1.81', '1.95'], 'Smiles': ['COc1ccc(C2c3ccc4ccccc4c3OC3NC(=S)NC(=S)C32)cc1', 'CO[C@]1(C)C[C@H](O[C@H]2[C@H](C)[C@@H](O[C@@H]3O[C@H](C)C[C@H](N(C)C)[C@H]3O)[C@](C)(OC)C[C@@H](C)[C@H](O)[C@H](C)CN(C)[C@H](Cn3cc(-c4ccc(F)cc4)nn3)COC(=O)[C@@H]2C)O[C@@H](C)[C@@H]1O', 'O=c1ccc2ccc(F)c3c2n1CC3CN1CCC(NCc2cc3c(cn2)OCCO3)CC1', 'COc1ccc2nccc([C@H](O)[C@@H](O)[C@H]3CC[C@H](NCc4ccc5c(n4)NC(=O)CS5)CO3)c2n1', 'Cc1cc(=O)cc(Cc2cc(O)cc(O)c2)o1'], 'Standard Type': ['MIC', 'MIC', 'MIC', 'MIC', 'MIC'], 'Standard Relation': ["'='", "'>'", "'='", "'='", "'>'"], 'Standard Value': [500.0, 128.0, 0.001, 0.001, 64.0], 'Standard Units': ['ug.mL-1', 'ug.mL-1', 'ug.mL-1', 'ug.mL-1', 'ug.mL-1'], 'pChEMBL Value': [nan, nan, nan, nan, nan], 'Data Validity Comment': [nan, nan, nan, nan, nan], 'Comment': [nan, nan, nan, nan, nan], 'Uo Units': ['UO_0000274', 'UO_0000274', 'UO_0000274', 'UO_0000274', 'UO_0000274'], 'Ligand Efficiency BEI': [nan, nan, nan, nan, nan], 'Ligand Efficiency LE': [nan, nan, nan, nan, nan], 'Ligand Efficiency LLE': [nan, nan, nan, nan, nan], 'Ligand Efficiency SEI': [nan, nan, nan, nan, nan], 'Potential Duplicate': [False, False, False, False, False], 'Assay ChEMBL ID': ['CHEMBL1286297', 'CHEMBL4336176', 'CHEMBL2426903', 'CHEMBL2426903', 'CHEMBL3088902'], 'Assay Description': ['Antimicrobial activity against Escherichia coli ATCC 25922 after 24 hrs by serial dilution method', 'Antibacterial activity against penicillin-susceptible Escherichia coli ATCC 25922 incubated for 24 hrs by broth dilution method', 'Antibacterial activity against wild type Escherichia coli ATCC 25922 assessed as growth inhibition by broth microdilution method', 'Antibacterial activity against wild type Escherichia coli ATCC 25922 assessed as growth inhibition by broth microdilution method', 'Antibacterial activity against Escherichia coli ATCC 25922 assessed as growth inhibition by broth microdilution method'], 'Assay Type': ['F', 'F', 'F', 'F', 'F'], 'BAO Format ID': ['BAO_0000218', 'BAO_0000218', 'BAO_0000218', 'BAO_0000218', 'BAO_0000218'], 'BAO Label': ['organism-based format', 'organism-based format', 'organism-based format', 'organism-based format', 'organism-based format'], 'Assay Organism': ['Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli'], 'Assay Tissue ChEMBL ID': ['None', 'None', 'None', 'None', 'None'], 'Assay Tissue Name': ['None', 'None', 'None', 'None', 'None'], 'Assay Cell Type': ['None', 'None', 'None', 'None', 'None'], 'Assay Subcellular Fraction': ['None', 'None', 'None', 'None', 'None'], 'Assay Parameters': [nan, nan, nan, nan, nan], 'Assay Variant Accession': [nan, nan, nan, nan, nan], 'Assay Variant Mutation': [nan, nan, nan, nan, nan], 'Target ChEMBL ID': ['CHEMBL354', 'CHEMBL354', 'CHEMBL354', 'CHEMBL354', 'CHEMBL354'], 'Target Name': ['Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli'], 'Target Organism': ['Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli'], 'Target Type': ['ORGANISM', 'ORGANISM', 'ORGANISM', 'ORGANISM', 'ORGANISM'], 'Document ChEMBL ID': ['CHEMBL1275291', 'CHEMBL4334458', 'CHEMBL2424615', 'CHEMBL2424615', 'CHEMBL3085737'], 'Source ID': [1, 1, 1, 1, 1], 'Source Description': ['Scientific Literature', 'Scientific Literature', 'Scientific Literature', 'Scientific Literature', 'Scientific Literature'], 'Document Journal': ['Eur. J. Med. Chem.', 'Eur J Med Chem', 'J. Med. Chem.', 'J. Med. Chem.', 'J. Nat. Prod.'], 'Document Year': [2010.0, 2019.0, 2013.0, 2013.0, 2013.0], 'Cell ChEMBL ID': ['None', 'None', 'None', 'None', 'None'], 'Properties': [nan, nan, nan, nan, nan], 'Mol': [ < rdkit.Chem.rdchem.Mol object at 0x7f840af07e90 > , < rdkit.Chem.rdchem.Mol object at 0x7f840af0c210 > , < rdkit.Chem.rdchem.Mol object at 0x7f841b2f6f30 > , < rdkit.Chem.rdchem.Mol object at 0x7f841b2f7530 > , < rdkit.Chem.rdchem.Mol object at 0x7f841b326a80 > ], 'CircularFingerprint': [array([[0., 0., 0., ..., 0., 0., 0.]]), array([[0., 0., 0., ..., 0., 0., 0.]]), array([[0., 0., 0., ..., 0., 0., 0.]]), array([[0., 1., 0., ..., 0., 0., 0.]]), array([[0., 0., 0., ..., 0., 0., 0.]])], 'ConvMolFeaturizer': [array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f84030cea50 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f84033f2e50 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f83fbe8cd10 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f83fbf442d0 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f8400690c10 > ],
        dtype=object)]}

    example_df = pd.DataFrame.from_dict(example_dict)

    dataset, csv = dataprep.convert_to_dataset(
        dataframe=example_df, X_col='CircularFingerprint', y_col='Standard Value', w_col=None, id_col=None, return_csv=True)

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

    # create example dataframe (values extracted randomly from existing ChEMBL query)
    example_dict = {'index': [4829, 5079, 731, 749, 3488], 'Molecule ChEMBL ID': ['CHEMBL1276627', 'CHEMBL4457967', 'CHEMBL2424894', 'CHEMBL2424889', 'CHEMBL3086842'], 'Molecule Name': [nan, nan, nan, nan, nan], 'Molecule Max Phase': [0, 0, 0, 0, 0], 'Molecular Weight': ['406.53', '880.11', '450.51', '497.58', '232.24'], '#RO5 Violations': ['0', '2', '0', '0', '0'], 'AlogP': ['4.12', '3.74', '2.66', '1.81', '1.95'], 'Smiles': ['COc1ccc(C2c3ccc4ccccc4c3OC3NC(=S)NC(=S)C32)cc1', 'CO[C@]1(C)C[C@H](O[C@H]2[C@H](C)[C@@H](O[C@@H]3O[C@H](C)C[C@H](N(C)C)[C@H]3O)[C@](C)(OC)C[C@@H](C)[C@H](O)[C@H](C)CN(C)[C@H](Cn3cc(-c4ccc(F)cc4)nn3)COC(=O)[C@@H]2C)O[C@@H](C)[C@@H]1O', 'O=c1ccc2ccc(F)c3c2n1CC3CN1CCC(NCc2cc3c(cn2)OCCO3)CC1', 'COc1ccc2nccc([C@H](O)[C@@H](O)[C@H]3CC[C@H](NCc4ccc5c(n4)NC(=O)CS5)CO3)c2n1', 'Cc1cc(=O)cc(Cc2cc(O)cc(O)c2)o1'], 'Standard Type': ['MIC', 'MIC', 'MIC', 'MIC', 'MIC'], 'Standard Relation': ["'='", "'>'", "'='", "'='", "'>'"], 'Standard Value': [500.0, 128.0, 0.001, 0.001, 64.0], 'Standard Units': ['ug.mL-1', 'ug.mL-1', 'ug.mL-1', 'ug.mL-1', 'ug.mL-1'], 'pChEMBL Value': [nan, nan, nan, nan, nan], 'Data Validity Comment': [nan, nan, nan, nan, nan], 'Comment': [nan, nan, nan, nan, nan], 'Uo Units': ['UO_0000274', 'UO_0000274', 'UO_0000274', 'UO_0000274', 'UO_0000274'], 'Ligand Efficiency BEI': [nan, nan, nan, nan, nan], 'Ligand Efficiency LE': [nan, nan, nan, nan, nan], 'Ligand Efficiency LLE': [nan, nan, nan, nan, nan], 'Ligand Efficiency SEI': [nan, nan, nan, nan, nan], 'Potential Duplicate': [False, False, False, False, False], 'Assay ChEMBL ID': ['CHEMBL1286297', 'CHEMBL4336176', 'CHEMBL2426903', 'CHEMBL2426903', 'CHEMBL3088902'], 'Assay Description': ['Antimicrobial activity against Escherichia coli ATCC 25922 after 24 hrs by serial dilution method', 'Antibacterial activity against penicillin-susceptible Escherichia coli ATCC 25922 incubated for 24 hrs by broth dilution method', 'Antibacterial activity against wild type Escherichia coli ATCC 25922 assessed as growth inhibition by broth microdilution method', 'Antibacterial activity against wild type Escherichia coli ATCC 25922 assessed as growth inhibition by broth microdilution method', 'Antibacterial activity against Escherichia coli ATCC 25922 assessed as growth inhibition by broth microdilution method'], 'Assay Type': ['F', 'F', 'F', 'F', 'F'], 'BAO Format ID': ['BAO_0000218', 'BAO_0000218', 'BAO_0000218', 'BAO_0000218', 'BAO_0000218'], 'BAO Label': ['organism-based format', 'organism-based format', 'organism-based format', 'organism-based format', 'organism-based format'], 'Assay Organism': ['Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli'], 'Assay Tissue ChEMBL ID': ['None', 'None', 'None', 'None', 'None'], 'Assay Tissue Name': ['None', 'None', 'None', 'None', 'None'], 'Assay Cell Type': ['None', 'None', 'None', 'None', 'None'], 'Assay Subcellular Fraction': ['None', 'None', 'None', 'None', 'None'], 'Assay Parameters': [nan, nan, nan, nan, nan], 'Assay Variant Accession': [nan, nan, nan, nan, nan], 'Assay Variant Mutation': [nan, nan, nan, nan, nan], 'Target ChEMBL ID': ['CHEMBL354', 'CHEMBL354', 'CHEMBL354', 'CHEMBL354', 'CHEMBL354'], 'Target Name': ['Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli'], 'Target Organism': ['Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli'], 'Target Type': ['ORGANISM', 'ORGANISM', 'ORGANISM', 'ORGANISM', 'ORGANISM'], 'Document ChEMBL ID': ['CHEMBL1275291', 'CHEMBL4334458', 'CHEMBL2424615', 'CHEMBL2424615', 'CHEMBL3085737'], 'Source ID': [1, 1, 1, 1, 1], 'Source Description': ['Scientific Literature', 'Scientific Literature', 'Scientific Literature', 'Scientific Literature', 'Scientific Literature'], 'Document Journal': ['Eur. J. Med. Chem.', 'Eur J Med Chem', 'J. Med. Chem.', 'J. Med. Chem.', 'J. Nat. Prod.'], 'Document Year': [2010.0, 2019.0, 2013.0, 2013.0, 2013.0], 'Cell ChEMBL ID': ['None', 'None', 'None', 'None', 'None'], 'Properties': [nan, nan, nan, nan, nan], 'Mol': [ < rdkit.Chem.rdchem.Mol object at 0x7f840af07e90 > , < rdkit.Chem.rdchem.Mol object at 0x7f840af0c210 > , < rdkit.Chem.rdchem.Mol object at 0x7f841b2f6f30 > , < rdkit.Chem.rdchem.Mol object at 0x7f841b2f7530 > , < rdkit.Chem.rdchem.Mol object at 0x7f841b326a80 > ], 'CircularFingerprint': [array([[0., 0., 0., ..., 0., 0., 0.]]), array([[0., 0., 0., ..., 0., 0., 0.]]), array([[0., 0., 0., ..., 0., 0., 0.]]), array([[0., 1., 0., ..., 0., 0., 0.]]), array([[0., 0., 0., ..., 0., 0., 0.]])], 'ConvMolFeaturizer': [array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f84030cea50 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f84033f2e50 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f83fbe8cd10 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f83fbf442d0 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f8400690c10 > ],
        dtype=object)]}

    example_df = pd.DataFrame.from_dict(example_dict)

    dataset, csv = dataprep.convert_to_dataset(
        dataframe=example_df, X_col='CircularFingerprint', y_col='Standard Value', w_col=None, id_col=None, return_csv=False)

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

    # create example dataframe (values extracted randomly from existing ChEMBL query)
    example_dict = {'index': [4829, 5079, 731, 749, 3488], 'Molecule ChEMBL ID': ['CHEMBL1276627', 'CHEMBL4457967', 'CHEMBL2424894', 'CHEMBL2424889', 'CHEMBL3086842'], 'Molecule Name': [nan, nan, nan, nan, nan], 'Molecule Max Phase': [0, 0, 0, 0, 0], 'Molecular Weight': ['406.53', '880.11', '450.51', '497.58', '232.24'], '#RO5 Violations': ['0', '2', '0', '0', '0'], 'AlogP': ['4.12', '3.74', '2.66', '1.81', '1.95'], 'Smiles': ['COc1ccc(C2c3ccc4ccccc4c3OC3NC(=S)NC(=S)C32)cc1', 'CO[C@]1(C)C[C@H](O[C@H]2[C@H](C)[C@@H](O[C@@H]3O[C@H](C)C[C@H](N(C)C)[C@H]3O)[C@](C)(OC)C[C@@H](C)[C@H](O)[C@H](C)CN(C)[C@H](Cn3cc(-c4ccc(F)cc4)nn3)COC(=O)[C@@H]2C)O[C@@H](C)[C@@H]1O', 'O=c1ccc2ccc(F)c3c2n1CC3CN1CCC(NCc2cc3c(cn2)OCCO3)CC1', 'COc1ccc2nccc([C@H](O)[C@@H](O)[C@H]3CC[C@H](NCc4ccc5c(n4)NC(=O)CS5)CO3)c2n1', 'Cc1cc(=O)cc(Cc2cc(O)cc(O)c2)o1'], 'Standard Type': ['MIC', 'MIC', 'MIC', 'MIC', 'MIC'], 'Standard Relation': ["'='", "'>'", "'='", "'='", "'>'"], 'Standard Value': [500.0, 128.0, 0.001, 0.001, 64.0], 'Standard Units': ['ug.mL-1', 'ug.mL-1', 'ug.mL-1', 'ug.mL-1', 'ug.mL-1'], 'pChEMBL Value': [nan, nan, nan, nan, nan], 'Data Validity Comment': [nan, nan, nan, nan, nan], 'Comment': [nan, nan, nan, nan, nan], 'Uo Units': ['UO_0000274', 'UO_0000274', 'UO_0000274', 'UO_0000274', 'UO_0000274'], 'Ligand Efficiency BEI': [nan, nan, nan, nan, nan], 'Ligand Efficiency LE': [nan, nan, nan, nan, nan], 'Ligand Efficiency LLE': [nan, nan, nan, nan, nan], 'Ligand Efficiency SEI': [nan, nan, nan, nan, nan], 'Potential Duplicate': [False, False, False, False, False], 'Assay ChEMBL ID': ['CHEMBL1286297', 'CHEMBL4336176', 'CHEMBL2426903', 'CHEMBL2426903', 'CHEMBL3088902'], 'Assay Description': ['Antimicrobial activity against Escherichia coli ATCC 25922 after 24 hrs by serial dilution method', 'Antibacterial activity against penicillin-susceptible Escherichia coli ATCC 25922 incubated for 24 hrs by broth dilution method', 'Antibacterial activity against wild type Escherichia coli ATCC 25922 assessed as growth inhibition by broth microdilution method', 'Antibacterial activity against wild type Escherichia coli ATCC 25922 assessed as growth inhibition by broth microdilution method', 'Antibacterial activity against Escherichia coli ATCC 25922 assessed as growth inhibition by broth microdilution method'], 'Assay Type': ['F', 'F', 'F', 'F', 'F'], 'BAO Format ID': ['BAO_0000218', 'BAO_0000218', 'BAO_0000218', 'BAO_0000218', 'BAO_0000218'], 'BAO Label': ['organism-based format', 'organism-based format', 'organism-based format', 'organism-based format', 'organism-based format'], 'Assay Organism': ['Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli'], 'Assay Tissue ChEMBL ID': ['None', 'None', 'None', 'None', 'None'], 'Assay Tissue Name': ['None', 'None', 'None', 'None', 'None'], 'Assay Cell Type': ['None', 'None', 'None', 'None', 'None'], 'Assay Subcellular Fraction': ['None', 'None', 'None', 'None', 'None'], 'Assay Parameters': [nan, nan, nan, nan, nan], 'Assay Variant Accession': [nan, nan, nan, nan, nan], 'Assay Variant Mutation': [nan, nan, nan, nan, nan], 'Target ChEMBL ID': ['CHEMBL354', 'CHEMBL354', 'CHEMBL354', 'CHEMBL354', 'CHEMBL354'], 'Target Name': ['Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli'], 'Target Organism': ['Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli', 'Escherichia coli'], 'Target Type': ['ORGANISM', 'ORGANISM', 'ORGANISM', 'ORGANISM', 'ORGANISM'], 'Document ChEMBL ID': ['CHEMBL1275291', 'CHEMBL4334458', 'CHEMBL2424615', 'CHEMBL2424615', 'CHEMBL3085737'], 'Source ID': [1, 1, 1, 1, 1], 'Source Description': ['Scientific Literature', 'Scientific Literature', 'Scientific Literature', 'Scientific Literature', 'Scientific Literature'], 'Document Journal': ['Eur. J. Med. Chem.', 'Eur J Med Chem', 'J. Med. Chem.', 'J. Med. Chem.', 'J. Nat. Prod.'], 'Document Year': [2010.0, 2019.0, 2013.0, 2013.0, 2013.0], 'Cell ChEMBL ID': ['None', 'None', 'None', 'None', 'None'], 'Properties': [nan, nan, nan, nan, nan], 'Mol': [ < rdkit.Chem.rdchem.Mol object at 0x7f840af07e90 > , < rdkit.Chem.rdchem.Mol object at 0x7f840af0c210 > , < rdkit.Chem.rdchem.Mol object at 0x7f841b2f6f30 > , < rdkit.Chem.rdchem.Mol object at 0x7f841b2f7530 > , < rdkit.Chem.rdchem.Mol object at 0x7f841b326a80 > ], 'CircularFingerprint': [array([[0., 0., 0., ..., 0., 0., 0.]]), array([[0., 0., 0., ..., 0., 0., 0.]]), array([[0., 0., 0., ..., 0., 0., 0.]]), array([[0., 1., 0., ..., 0., 0., 0.]]), array([[0., 0., 0., ..., 0., 0., 0.]])], 'ConvMolFeaturizer': [array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f84030cea50 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f84033f2e50 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f83fbe8cd10 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f83fbf442d0 > ],
        dtype=object), array([ < deepchem.feat.mol_graphs.ConvMol object at 0x7f8400690c10 > ],
        dtype=object)]}

    example_df = pd.DataFrame.from_dict(example_dict)

    dataset, csv = dataprep.convert_to_dataset(
        dataframe=example_df, X_col='CircularFingerprint', y_col='Standard Value', w_col=None, id_col=None, return_csv=False)

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
