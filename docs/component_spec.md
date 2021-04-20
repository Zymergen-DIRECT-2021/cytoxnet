# Cytoxnet Component Specification
The package is intended as an exploration and a toolbox for the prediction of
cytotoxicity of compounds, but aims to leverage datasets of other targets
for transfer learning.
***
***
## package format
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
***

## `cytoxnet/dataprep/`

### `cytoxnet/dataprep/io.py`

```
load_file(datafile: str,
          cols: str or list of str = None) -> pandas DataFrame
```
- _Use Case_: (1)
- _Inputs_: File location string, columns to keep
- _Outputs_: Dataframe of dataset on file

```
load_XXX(prepare: bool = False)
```
- _Use Case_: (1)
- _Inputs_: Whether or not to featurize and prepare set using a default
  pipeline
- _Outputs_: Dataframe of package dataset

### `cytoxnet/dataprep/featurize.py`

```
clean_dataframe(dataframe,
                cols: str or list of str = None,
                duplicates: str = 'drop',
                nans: str = 'drop') -> pandas DataFrame
```
- _Use Case_: (1, 2)
- _Inputs_: Dataframe, columns to clean, how to handle nans and duplicates.
- _Outputs_: Dataframe

```
featurize(dataframe,
          target_cols: list of str,
          id_col: str = 'smiles',
          featurizer: str = 'CircularFingerprint',
          descriptor_cols: list of str = None) -> dataframe
```
- _Use Case_: (2)
- _Inputs_: Dataframe, columns containing targets and features, type of
  featurizer to use, additional descriptors to keep as features.
- _Outputs_: Dataframe with X and y columns
- *Temporary note*: call convert to catagorical on descriptor cols and target

### `cytoxnet/dataprep/dataprep.py`

```
convert_to_catagorical(dataframe,
                       cols: list of str)
```
- _Use Case_: (3)
- _Inputs_: Dataframe, columns to consider for catagorical conversion
- _Outputs_: Dataframe with considered columns that are non numeric converted.

```
convert_to_dataset(dataframe,
                   X_col: str = 'X',
                   y_col: str = 'y',
                   w_col: str = None)
```
- _Use Case_: (3)
- _Inputs_: Dataframe, non default X and y columns
- _Outputs_: deepchem dataset from dataframe

```
preprocess(dataset,
           transformations: list of str = ['NormTransform',],
           splitter: str = 'RandomSplitter')
```
- _Use Case_: (3)
- _Inputs_: Dataset and transformations to make.
- _Outputs_: Dataset(s) preprocessed and ready for ml

```
pipeline(datafile: str,
         id_cols: list of str,
         descriptor_cols: list of str,
         target_cols: list of str,
         **kwargs)
```
- _Use Case_: (1,2,3)
- _Inputs_: Datafile location, columns in question, keywords for pipeline.
- _Outputs_: Dataset(s) preprocessed and ready for ml.

### `cytoxnet/dataprep/analyze.py`

```
dataset_transferability(dataset1, dataset2)
```
- _Use Case_: (9)
- _Inputs_: Datasets to compared.
- _Outputs_: Metrics and visuals of the transferability of dataset 1 to 2.

## `cytoxnet/models/`

### `cytoxnet/models/models.py`

```
ToxModel.help(model_name: str)
```
- _Use Case_: (4.1)
- _Inputs_: Model name, model method or None
- _Outputs_: Printout of help on model, model method, or available models.


```
ToxModel(model_name: str, *kwargs)
```
- _Use Case_: (4.2)
- _Inputs_: Model name, keyword arguments for that model.
- _Outputs_: Instance of ToxModel with model type.

Use cases 4.3+ and 6 are handled by model instance. See `deepchem.models.Model`.

```
ToxModel.pretrain(dataset, fix: bool: False)
```
- _Use Case_: (7)
- _Inputs_: dataset to pretrain on
- _Outputs_: Trained model with output ready for new target and potentially
  fixed layers
  
```
ToxModel.transfer(dataset, fix: bool: False)
```
- _Use Case_: (7)
- _Inputs_: dataset to transfer to
- _Outputs_: Trained model on new targets, with layers potentially fixed to
  previous training.
  
### `cytoxnet/models/opt.py`

```
HypOpt(model_name: str,
       search_type: str,
       searchable_params: dict,
       fixed_params: dict)
```
- _Use Case_: (5)
- _Inputs_: Model class type, hyperparameter search scheme and space
- _Outputs_: Optimized model and best parameters
  
### `cytoxnet/models/analyze.py`

```
ModelAnalysis(tox_model, dataset)
```
- _Use Case_: (8)
- _Inputs_: Test dataset and a trained model
- _Outputs_: Metrics and visuals of model performance.