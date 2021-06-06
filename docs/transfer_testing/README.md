# Transfer learning testing
Graph Neural Networks were lightly investigated as an alternative to multitask learning in order to leverage foreign datasets. This was only conducted for regression, but should also be tested for classification. Hyperparemeter optimization was conducted for a single target graph neural network training on Zhu's rat dataset (see data reports). TPE sampler was used to conduct the search, see [optuna](https://optuna.readthedocs.io/en/stable/reference/samplers.html). The indepepndant algea test set was used here to allow for comparison with the multitask models.

For the best hyperparemeters identified, the following models were tested:
- Train on algea data alone for 100 epochs, test on algea test set
- Train on rat data for 200 epochs, fix all but the last dense regression layer, and then train on algea data for 100 epochs and test on the algea test set.

Pretraining on rat data marginally improved performance over the baseline graph model, however it did not do better than the baseline RFR.

## Contents
- `create_studies.py`: prepares an optuna study for hyperparameter optimization
- `graph_regression.py`: Runs hyperparameter optimization on a single tast graph model
- `transfer_testing.ipynb`: for the best hyperparmeters, evaluate transfer learning on the test set.