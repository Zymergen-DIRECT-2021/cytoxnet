# Screening of model and feature types
This directory contains scripts used to screen model and feature types. Cross validation is used in each case. No model parameters are passed, instead using models out of the box - this is a first pass attempt to narrow the model and feature scope, and hyperparameter opt is conducted on the results of these scripts (see multitask_evalution directory)

## Contents

- `evaluate_graph_class.py`: evaluates graph CNNs for classification tasks
- `evaluate_grid_class.py`: runs a grid search on a few model types (GPC, RFC, KNNC) and feature types from deepchem (CircularFingerprint, RDKitDescriptors, Mordred)
- `evaluate_graph_regr.py`: evaluates graph CNNs for regression tasks
- `evaluate_grid_regr.py`: runs a grid search on a few model types (GPR, RFR, LASSO) and feature types from deepchem (CircularFingerprint, RDKitDescriptors, Mordred)

These scripts create csv files of the results.

- `plot_results.ipynb`: visualization of the search