# Cytoxnet Component Specification
The package is intended as a dataset exploration and a toolbox for the
prediction of cytotoxicity of compounds, but aims to leverage datasets of other
targets for transfer learning.
***
***
## package format
```
/cytoxnet
|-/data
|-/dataprep
| |-io.py
| |-featurize.py
| |-dataprep.py
|-/models
| |-models.py
| |-evaluate.py
| |-analyze.py
|-/tests
```
***
## `cytoxnet/data/`

**Included Data**
ChemBL Ecoli MIC
Lunghini Algae EC50
Lunghini Daphnia EC50
Lunghini Fish LC50
Lunghini Combined
Zhu Rat LD50
!!!! Temporary note-add citations and info!!!!

## `cytoxnet/dataprep/`

### `cytoxnet/dataprep/io.py`

```
load_data(datafile: str,
          cols: str or list of str = None,
          id_cols: str or list of str = None,
          duplicates: str = 'drop',
          nans: str = 'keep') -> pandas DataFrame
```
- __Use Case__: (1)
- __Inputs__: File location string (csv)
- __Optional Inputs__:  Columns to keep (default None keeps all). Columns to be used to handle duplicates and nans (default None keeps all). Indication of whether to 'keep' or 'drop' nans (default 'keep'). Indication of whether to 'keep' or 'drop' duplicates (default 'drop').  
- __Outputs__: Dataframe of the dataset in the input file.
- __Summary__: The user provides a path to a csv file and can optionally choose to specify columns the they would like to keep. The user can also specify whether to keep or drop nans and which columns they would like to use to do so by providing id_cols. The function removes nans and duplicates based on specified id_cols and only keeps specified columns if desired.. The function returns a dataframe of the data contained in the specified csv, potentially with unwanted columns removed and rows removed based on nans and duplicates.

```
create_compound_codex(db_path: str = './database',
                          id_col: str = 'smiles',
                          featurizers: str or list of str = None,
                          **kwargs) --> compounds codex (compounds.csv)
```
- __Use Case__: (1)
- __Optional Inputs__: Path to folder where dababase files will be located, default './database'. Column ID corresponding to compound identity, default 'smiles'. Desired featurizers to include in the initialized compounds codex, default None.
- __Outputs__: Does not return any variable, crates a compound codex at a specified location. 
- __Summary__: All inputs are optional. By default, calling the function will create a folder or use the folder './database' and create a compounds.csv file at that location. The codex can be used to track canonicalized smiles and corresponding features.  

```
add_datasets(dataframes: df or str or list of df or str,
                 names: str or list of str,
                 id_col: str = 'smiles',
                 db_path: str = './database',
                 new_featurizers: str or list of str = None,
                 **kwargs) 
```
- __Use Case__: (1)
- __Inputs__: One or more datasets specified as path(s) to csv(s) or dataframe(s); can also specify names of the datasets that are already included in package data. Name(s) of the dataset(s) passed.
- __Optional Inputs__: Column representing the identity of the compound in all datasets, default 'smiles'. The path to the folder where database files are located, default './database'. The new featurization methods that should be used to featurze the new, as well as existing data.
- __Outputs__: Returns a dictionary of cleaned dataframes identified via corresponding dataset names. 
- __Summary__: The user specifies dataframe(s) or csv(s) or existing package data name(s) and their correspnding names and the function canonicalizes the compound id column (e.g. 'smiles' column), adds the compounds' canonicalized smiles to the compounds codex, and adds computes and stores desired features for the new and the previous dataset. The output dictionary can be used to construct the database.

### `cytoxnet/dataprep/featurize.py`

```
molstr_to_Mol(dataframe, id_col='smiles')
```
- __Use Case__: (1, 2)
- __Inputs__: A dataframe containing InChI or SMILES representations of molecules and corresponding column name identifying the column containing them.
- __Optional Inputs__: Column identifier, defaults to 'smiles'.
- __Outputs__: Dataframe with an additional column containing compound as RDKit Mol Objects. 
- __Summary__: The user inputs a datagframe with InChI or SMILES representations of molecules and the name of the column it it is not 'smiles'. The function checks the column name and if it contains the letters 'inchi' or 'smiles', it converts each string representation to an RDKit Mol object and adds a 'Mol' column to the dataframe containing the Mol representations of the molecules.

``` 
from_np_array(array_string: str)
```
- __Use Case__: (2)
- __Inputs__: A string containing an array.
- __Outputs__: A numpy array.
- __Summary__: User inputs and array that is in string form. The function converts the string into a numpy array, which is returned.

```
add_features(dataframe: df,
             id_col: str = 'smiles',
             method: str = 'CircularFingerprint',
             codex: str = None,
             canonicalize: bool = True,
             drop_na: bool = True,
             **kwargs) -> dataframe
```
- __Use Case__: (2)
- __Inputs__: Dataframe containing a column with string representations of molecules. Identification of tha column name if not 'smiles'.
- __Optional Inputs__: Compound ID column name, default 'smiles'. Featurization method, default 'CircularFingerprint'. Path to a compounds codex containing compound IDs and features, default . Indication of whether or not to canonicalize the SMILES, default True. Indication of whether or not to drop nans, default True.
- __Outputs__: Dataframe containing a column with a column containing the desired features. If codex is specified, it adds compounds and features for those not yet in the codex.
- __Summary__: The user inputs a dataframe with a string representation of a molecule and the corresponding name of that column (if not 'smiles'). The function first converts string representations to Mol objects using the molstr_to_Mol function. It then retrieves the desired featurization object type from deepchem and computes features for the compounds in the dataframe. A dataframe with an additional column containing the desired features is returned. If a codex is provided, the function adds any compounds not yet present and their corresponding features to the codex. The function also canonicalizes the compound identifier by default and drops rows containing nans in the ID column.

### `cytoxnet/dataprep/dataprep.py`

```
convert_to_categorical(dataframe,
                       cols: list of str = None) --> dataframe
```
- __Use Case__: (3)
- __Inputs__: Dataframe and identification of columns to consider for categorical conversion (default None).
- __Outputs__: Dataframe with considered columns that are non-numeric converted to integer values.
- __Summary__: The user inputs a dataframe with a column containing non-numeric categorical data and the name of that column. The function converts that column to integer values.

```
binarize_targets(dataframe,
                 target_cols: Union[str, List[str]],
                 high_positive: bool = False,
                 percentile: float = 0.5,
                 value: Union[float, List[float]] = None)
```
- __Use Case__: (3)
- __Inputs__: DataFrame with a target column(s) to be binarized. Name of the column to be binarized. 
- __Optional_Inputs__: Option to specify whether values above a threshold are True or False (default False, e.g. above the defined threshhold, values for toxicity are set to False to correspond to not toxic). A percentile value indicating the relative position of the threshold based on the distribution of the data, default 0.5. Value(s) to use as a threshold.
- __Outputs__: 
- __Summary__: 

```
convert_to_dataset(dataframe,
                   X_col: str = 'X',
                   y_col: str = 'y',
                   w_col: str = None,
                   id_col: str = None)
```
- __Use Case__: (3)
- __Inputs__: Dataframe, non default X and y columns
- __Outputs__: deepchem dataset from dataframe

```
preprocess(dataset,
           transformations: list of str = ['NormTransform',],
           splitter: str = 'RandomSplitter', **kwargs)
```
- __Use Case__: (3)
- __Inputs__: Dataset and transformations to make.
- __Outputs__: Dataset(s) preprocessed and ready for ml

```
pipeline(datafile: str,
         id_cols: list of str,
         descriptor_cols: list of str,
         target_cols: list of str,
         **kwargs)
```
- __Use Case__: (1,2,3)
- __Inputs__: Datafile location, columns in question, keywords for pipeline.
- __Outputs__: Dataset(s) preprocessed and ready for ml.

### `cytoxnet/dataprep/analyze.py`

```
dataset_transferability(dataset1, dataset2)
```
- __Use Case__: (9)
- __Inputs__: Datasets to be compared.
- __Outputs__: Metrics and visuals of the transferability of dataset 1 to 2.

## `cytoxnet/models/`

### `cytoxnet/models/models.py`

```
ToxModel.help(model_name: str)
```
- __Use Case__: (4.1)
- __Inputs__: Model name, model method or None
- __Outputs__: Printout of help on model, model method, or available models.


```
ToxModel(model_name: str, *kwargs)
```
- __Use Case__: (4.2)
- __Inputs__: Model name, keyword arguments for that model.
- __Outputs__: Instance of ToxModel with model type.

Use cases 4.3+ and 6 are handled by model instance. See `deepchem.models.Model`.

```
ToxModel.pretrain(dataset, fix: bool: False)
```
- __Use Case__: (7)
- __Inputs__: dataset to pretrain on
- __Outputs__: Trained model with output ready for new target and potentially
  fixed layers
  
```
ToxModel.transfer(dataset, fix: bool: False)
```
- __Use Case__: (7)
- __Inputs__: dataset to transfer to
- __Outputs__: Trained model on new targets, with layers potentially fixed to
  previous training.
  
### `cytoxnet/models/opt.py`

```
HypOpt(model_name: str,
       search_type: str,
       searchable_params: dict,
       fixed_params: dict)
```
- __Use Case__: (5)
- __Inputs__: Model class type, hyperparameter search scheme and space
- __Outputs__: Optimized model and best parameters
  
### `cytoxnet/models/analyze.py`

```
ModelAnalysis(tox_model, dataset)
```
- __Use Case__: (8)
- __Inputs__: Test dataset and a trained model
- __Outputs__: Metrics and visuals of model performance.
