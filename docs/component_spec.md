# Cytoxnet Component Specification
The package is intended as an exploration and a toolbox for the prediction of
cytotoxicity of compounds, but aims to leverage datasets of other targets
for transfer learning.
***
***
## package format
```
/cytoxnet
|-/dataprep
| |-io.py
| |-featurize.py
| |-dataprep.py
| |-analyze.py
|-/models
| |-models.py
| |-opt.py
| |-analyze.py
|-/tests
```
***

## `cytoxnet/dataprep/`

### `cytoxnet/dataprep/io.py`

```
load_file(datafile: str,
          cols: str or list of str = None) -> pandas DataFrame
```
- __Use Case__: (1)
- __Inputs__: File location string, columns to keep
- __Outputs__: Dataframe of dataset on file

```
load_XXX(prepare: bool = False)
```
- __Use Case__: (1)
- __Inputs__: Whether or not to featurize and prepare set using a default
  pipeline
- __Outputs__: Dataframe of package dataset

### `cytoxnet/dataprep/featurize.py`

```
clean_dataframe(dataframe,
                cols: str or list of str = None,
                duplicates: str = 'drop',
                nans: str = 'drop') -> pandas DataFrame
```
- __Use Case__: (1, 2)
- __Inputs__: Dataframe, columns to clean, how to handle nans and duplicates.
- __Outputs__: Dataframe

```
featurize(dataframe,
          target_cols: list of str,
          id_col: str = 'smiles',
          featurizer: str = 'CircularFingerprint',
          descriptor_cols: list of str = None) -> dataframe
```
- __Use Case__: (2)
- __Inputs__: Dataframe, columns containing targets and features, type of
  featurizer to use, additional descriptors to keep as features.
- __Outputs__: Dataframe with X and y columns
- *Temporary note*: call convert to catagorical on descriptor cols and target

### `cytoxnet/dataprep/dataprep.py`

```
convert_to_catagorical(dataframe,
                       cols: list of str)
```
- __Use Case__: (3)
- __Inputs__: Dataframe, columns to consider for catagorical conversion
- __Outputs__: Dataframe with considered columns that are non numeric converted.

```
convert_to_dataset(dataframe,
                   X_col: str = 'X',
                   y_col: str = 'y',
                   w_col: str = None)
```
- __Use Case__: (3)
- __Inputs__: Dataframe, non default X and y columns
- __Outputs__: deepchem dataset from dataframe

```
preprocess(dataset,
           transformations: list of str = ['NormTransform',],
           splitter: str = 'RandomSplitter')
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
- __Inputs__: Datasets to compared.
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