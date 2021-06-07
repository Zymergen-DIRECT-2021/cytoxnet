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
| |-database.py
|-/models
| |-models.py
| |-evaluate.py
| |-analyze.py
|-/tests
```
***
## `cytoxnet/data/`

### Included Data

__ChemBL Ecoli MIC__
- Ecoli MIC query from ChemBL
- Nucleic Acids Res. 2019; 47(D1):D930-D940. doi: 10.1093/nar/gky1075
- See dataset report in docs/data_reports/chembl_ecoli.ipynb for more detail

__Lunghini__
- Algae, daphnia, and fish data from Lunghini et al., 2020 
- Lunghini, F, Marcou, G, Azam, P, Enrici, M.H, Van Miert, E, & Varnek, A. (2020). Consensus QSAR models estimating acute toxicity to aquatic organisms from different trophic levels: Algae, Daphnia and fish. SAR and QSAR in Environmental Research, 31(9), 655-675.
- See dataset report in docs/data_reports/lunghini_ecotoxological.ipynb for more detail

__Lunghini Algae EC50__
- Only algae data from Lunghini et al, 2020

__Lunghini Daphnia EC50__
- Only daphnia data from Lunghini et al., 2020

__Lunghini Fish LC50__
- Only fish data from Lunghini et al., 2020

__Zhu Rat LD50__
- Rat LD50 data from Zhu et al., 2009
- Zhu, Hao, et al. “Quantitative structure− activity relationship modeling of rat acute toxicity by oral exposure.” Chemical research in toxicology 22.12 (2009): 1913-1921.
- See dataset report in docs/data_reports/zhu_rat.ipynb for more detail

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
                          **kwargs) -> compounds codex (compounds.csv)
```
- __Use Case__: (10, 11)
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
- __Use Case__: (10, 11)
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
                       cols: list of str = None) -> dataframe
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
                 value: Union[float, List[float]] = None) -> dataframe
```
- __Use Case__: (3)
- __Inputs__: DataFrame with a target column(s) to be binarized. Name of the column to be binarized. 
- __Optional_Inputs__: Option to specify whether values above a threshold are True or False (default False, e.g. above the defined threshhold, values for toxicity are set to False to correspond to not toxic). A percentile value indicating the relative position of the threshold based on the distribution of the data, default 0.5. Value(s) to use as a threshold.
- __Outputs__: Dataframe containing the binarized data. 
- __Summary__: The user inputs a dataframe containing continuous data to be binarized and the corresponding names of the column(s). The user can optionally specify a percentile value to define the threshold based on the distribution (default 0.5) or can specify a value as a threshold. The user can choose if values above the threshold are considered True or False (default is False, i.e. above the threshold values are False, corresponding to non-toxic). 

```
canonicalize_smiles(smiles, raise_error=False)
```
- __Use Case__: (3)
- __Inputs__: A SMILES string. 
- __Optional_Inputs__: Whether to raise an error if canonicalizing fails (by default nan is returned)
- __Outputs__: Canonicalized smiles string.
- __Summary__: The user inputs a SMILES string that they would like to canonicalize, the function does so and returns the canonicalized SMILES string. The user can optionally specify whether failing to canonicalize a SMILES string returns an error or a nan. 

```
handle_sparsity(dataframe,
                    y_col: List[str],
                    w_label: str = 'w') -> dataframe
```
- __Use Case__: (3)
- __Inputs__: A dataframe containing sparse targets and the corresponding columns containing the targets. 
- __Optional_Inputs__: A string to add to target column names for the new weights columns (defauilt '_w'). 
- __Outputs__: DataFrame with target columns containing 0.0's in place of nans and with additional columns that have weights giving all the nan values weights of zero.
- __Summary__: The user inputs a dataframe with sparse data and corresponding column names. The function replaces nans with 0.0 so that the data can be input into a machine learning model and creates correspond weight columns that contain zero weights for the values that were originally nans. The name of the weight columns is by default 'w' + target column name , or the user can specify the string. 

```
convert_to_dataset(dataframe,
                   X_col: str = 'X',
                   y_col: str = 'y',
                   w_col: str = None,
                   w_label: str = None,
                   id_col: str = None) -> dataset
```
- __Use Case__: (3)
- __Inputs__: A dataframe containing feature and target columns and their corresponding column name.
- __Optional_Inputs__: Column names for X and Y (if not 'X' and 'y'). Name of weight column(s) (default None). Preceding label for weight columns (default None; see handle_sparsity function). The name of the column containing molecule identities. 
- __Outputs__: DeepChem dataset containing the X and y data and IDs, if provided. 
- __Summary__: The user passes a dataframe with at least feature and target columns and their corresponding column names. They can also choose to include and identify an ID column, if desired. The function converts the dataframe into a deepchem dataset object containing the passed data (and IDs, if passed) and returns the dataset object.

```
data_transformation(dataset,
                    transformations: list = ['NormalizationTransformer'],
                    to_transform: list = [],
                    **kwargs) -> transformed dataset, transformer objects list
```
- __Use Case__: (3)
- __Inputs__: A deepchem dataset object containing data. 
- __Optional_Inputs__: List of transformations to perform (default is only Normalization). List of elements to transform (i.e. 'X', 'y', or 'w'). Keyword arguments passed to the transformer object. 
- __Outputs__: The transformed dataset. List of transformer objects containign transformation information. 
- __Summary__: The user inputs a deepchem dataset object with data to be transformed and a list of desired transformations (default is just a normalization transformer). The function transforms the data by retreiving and applying the specified transformers from DeepChem and returns the transformed data as well as a list of the applied transform objects. 



```
data_splitting(dataset,
               splitter: str = 'RandomSplitter',
               split_type: str = 'train_valid_test_split',
               **kwargs) -> split dataset
```
- __Use Case__: (3)
- __Inputs__: A deepchem dataset object containing data. 
- __Optional_Inputs__: Desired splitter (default RandomSplitter). Desired split type (default train_valid_test_split). Keyword arguments passed to the splitter object. 
- __Outputs__: A set of dataset objects split based on the input data and splitter information. 
- __Summary__: The user passes a DeepChem dataset object containing data to be split and specifies the type of splitter and split type. The function retrieves and applies the specified splitter from DeepChem and returns the split dataset.

###`cytoxnet/dataprep/database.py` 

```
def drop_table(table_name: str) -> removes table from cytoxnet.db schema
```
- __Use Case__: (12)
- __Inputs__: Name of table to be dropped from SQLite database.
- __Outputs__: Drops table from SQLite database schema.
- __Summary__: Called by user or other function to drop a table of a given name if it exists in the cytoxnet.db schema.

```
table_creator(table_name: str, 
              dataframe, 
              codex: bool = False, 
              id_col: int = None) -> creates new database table from dataframe object
```
- __Use Case__: (10, 12)
- __Inputs__: Name to be given to new table. Dataframe object used to create new table.
- __Optional_Inputs__: True or False statement indicating whether the table being created/dataframe being input is the compound codex (see `cytoxnet.dataprep.io.create_compound_codex` function).  Integer specifying the position of an 'id' column (ie, column containing a unique int value for every row) if one exists.
- __Outputs__: New table within cytoxnet.db, with 'ids' column set as primary key and a foreign key linking each tuple to a tuple within the compounds codex table.
- __Summary__: User inputs the name of a table and a dataframe, and a table will be created (or updated if the specified table already exists) based on the input dataframe.

```
def tables_to_database(dataframe_dict) -> creates new database tables from dictionary of table names and dataframes
```
- __Use Case__: (10, 12)
- __Inputs__: Dictionary with names to be given to new tables as the keys, and dataframe objects used to create new tables as the values.
- __Outputs__: New tables within cytoxnet.db, with 'ids' columns set as primary keys and a foreign keys relating the tables to the compounds codex table.
- __Summary__: User inputs a dictionary object with names and dataframes, and for each name-dataframe pair, the `table_creator` function is called.

```
def query_to_dataframe(tables: str or list of str, 
                       features_list: str or list of str = None) -> creates new database table from dataframe object
```
- __Use Case__: (10, 12)
- __Inputs__: Name of table or tables to be included in query.
- __Optional_Inputs__: Name of stored feature type or types to include in the query.
- __Outputs__: Dataframe object based on the relational table generated from the query, containing all target, activity, compound, and (if specified) feature data from specified tables.
- __Summary__: User inputs the tables from which data is to be pulled, and a query is performed, retrieving all critical data from the specified datasets in the form of a dataframe object.

## `cytoxnet/models/`


### `cytoxnet/models/models.py`

```
class ToxModel 
```
- __Use Case__: (4)
- __Parameters__: model_name, the name of the model type to initialize. transformers, data transformations to apply to output predictions (for untransforming the data to raw space).  tasks, names for the different targets (Default only one unnamed task). use_weights (default False), only relevant for sklearn models thats can accept weights for fitting. Keyword arguments are passed to the model type for initialization.
- __Methods__: details below 


```
ToxModel._check_model_avail(model_name: str)
```
- __Inputs__: A model name. 
- __Outputs__: Prints out a statement including available models if the specified model is not available. 
- __Summary__: The user passes a model name and the method and the function prints information on available models if the model is not available. 

```
ToxModel._import_model_type(model_name: str)
```
- __Inputs__: A model name. 
- __Outputs__: A model class corresponding to the input model type. 
- __Summary__:  The user specifies the type of model and the function retreives and returns the corresponding model class. 


```
ToxModel.help(model_name: str)
```
- __Use Case__: (4)
- __Optional Inputs__: Model name (default None). 
- __Outputs__: Printout of list of models or details on a specific model. 
- __Summary__: User exectues this method to get information on models and, if a specific model is passed, can get detailed information on that model. 

```
ToxModel.evaluate(self,
                  dataset: Dataset,
                  metrics: List[Union[str, Metric]],
                  untransform: bool = False,
                  per_task_metrics: bool = False,
                  use_sample_weights: bool = False,
                  n_classes: int = None,
                  **kwargs) -> dict
```
- __Use Case__: (4)
- __Inputs__: DeepChem dataset object and list of desired metric(s). 
- __Optional Inputs__: Whether to untransform the the data (default False). Whether to use sample weights (default False). Number of classes (default None, only used for classification). For multitask, option to specify whether to calculate a metric for every task in the multitask model. 
- __Outputs__: A dictionary containing the metric score names and their corresponding values. Optinally, a dictionary containing the per-task metrics. 
- __Summary__: The user inputs a DeepChem dataset object and a list of metrics they would like the method to compute. The function evaluates the models and returns a dictionary containing the metrics. If used for a multitask model, the method can also optionally return a dictionary of per-task metrics. 

```
ToxModel.visualize(self,
                  viz_name: Union[str, object],
                  dataset: Dataset,
                  **kwargs) -> visual 
```
- __Use Case__: (8)
- __Inputs__:  Name of the visualization function to use. A DeepChem dataset object with the data to be visualized.
- __Optional Inputs__: Keyword arguments are passed to the visualization function. 
- __Outputs__: A visualization of the passed data.
- __Summary__: The user passes the DeepChem dataset they would like to vizualize and the function that they would like to use to do so. The method retreives the specified viasualization function and applies it to output the desired visual. 


```
ToxModel(model_name: str, *kwargs)
```
- __Use Case__: (4.2)
- __Inputs__: Model name, keyword arguments for that model.
- __Outputs__: Instance of ToxModel with model type.

Use cases 4.3+ and 6 are handled by model instance. See `deepchem.models.Model`.

```
ToxModel.pretrain(model: Model,
                  dataset: Union[Dataset, str],
                  **kwargs)
```
- __Use Case__: (7)
- __Inputs__: A model to be pretrained. DeepChem dataset to pretrain on or the string name of a dataset included in the package. 
- __Optional Inputs__: Keyword arguments passed to the model. 
- __Outputs__: Pretrained model with output ready for new target. 
- __Summary__: The user passes a model type and a dataset to use to pretrain the model. The method returns the pretrained model. 
  
```
ToxModel.transfer(model: Model,
                  dataset: Dataset,
                  **kwargs)
```
- __Use Case__: (7)
- __Inputs__: Model to conduct transfer learning on. Dataset with the task to transfer to. 
- __Outputs__: The transferred model.
- __Summary__: The user inputs a model to transfer on and a dataset to transfer to. The method fixes the already-learned inner layers of th model and trains the outer/pooling layers on the new task data. The new, transferred model is returned. 
  
### `cytoxnet/models/evaluate.py`

```
evaluate_crossval(datafile: Union[str, DataFrame],
                  ml_model: str,
                  feat_method: str,
                  target: Union[str, List[str]],
                  k: int = 5,
                  codex: str = None,
                  fit_kwargs: dict = {},
                  binary_percentile=None,
                  transformations=[],
                  to_transform=['y'],
                  **kwargs)
```
- __Use Case__: (5)
- __Inputs__: The file path, name in package, or dataframe to be used as the development set. The name of the ML model to test. The name of the deepchem featurizer to use. The column name corresponding to target(s) in the datafile.
- __Optional Inputs__: Number of folds for cross-valization (default 5). Path to the compound codex containing molecule IDs and their corresponding features. Binary percentile value to pass to the binarize_targets function (see dataprep.py module). Transformations to perform on the data. List of data column(s) to transform. 
- __Outputs__: Dictionary of metrics for corresponding feature type, model, and dataset. 
- __Summary__: User passes the development dataset, the model type to test, the featurizer type to use, and the target column name(s). The function evaluates the R2 and MSE (for a regression task) or the jaccard and recall score (for classification) and returns them as a dictionary with metrics that correspond to a given model, dataset, and feature type. The user can avoid recomputation of features, if already calculated, by passing the compounds codex that contains the features. 

```
grid_evaluate_crossval(datafiles: List[Union[str, DataFrame]],
                       ml_models: List[str],
                       feat_methods: List[str],
                       targets_codex: dict,
                       k: int = 5,
                       parallel: bool = True,
                      **kwargs)
```
- __Use Case__: (4, 5)
- __Inputs__: List of file path(s), name(s) in package, or dataframe(s) to be used as the development set. The name(s) of ML models to test. The name(s) of featurizer(s) to use. Dictionary of target column/datafile pairs. 
- __Optional Inputs__: Number of folds for cross-valization (default 5).  Keyword arguments passed to the evaluate_crossval function (see above)
- __Outputs__: Dictionary of metrics for corresponding feature type, model, and dataset. 
- __Summary__: User can pass multiple development datasets, model types, and featurizers to test. A dictionary of datafile/target column names is required. The function evaluates the R2 and MSE (for a regression task) or the jaccard and recall score (for classification) for all combinations of feature, model, and dataset, and returns them as a dictionary with metrics that correspond to a given model, dataset, and feature type. See evaluate_crossval function above for more details. 



### `cytoxnet/models/analyze.py`

```
pair_predict(model: object,
             dataset: Dataset,
             task: str = None,
             untransform: bool = True,
             return_df: bool = False) -> Viz
```
- __Use Case__: (8)
- __Inputs__: The ToxModel for the visualization. The testing dataset. 
- __Optional Inputs__: The task to use for plotting, if multitask (default None). Whether or not to untransform the data (default True). Whether or not to return the dataframe containing predictions and true values (default False). 
- __Outputs__: A pairwise regression prediction plot that plots the predicted values agains the true values and including an x=y line and a linear fit line. Optionally, also returns dataframe with corresponding true and predicted values. 
