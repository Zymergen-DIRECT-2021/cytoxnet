# Multitask learning evaluation
For the models/features screened in model screening:
- Random forests with RDKitDescriptors
- Graph Convolutional neural networks

Single target (baseline) and multitask learning was conducted. Random Forest multitask models were imputed both with mean imputation and [iterative interpolation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html). A set of algea data is held out as an independant test set for all cases. Extensive hyperparameter optimization is done for each model using TPE Sampling, see [optuna](https://optuna.readthedocs.io/en/stable/reference/samplers.html) on the remaining development data. See data_reports directory. 5 fold cross validation was used.

Contents
`/regression`
> contains scripts used for multiprocessor hyperparameter evaluation for regression models, and the results as .db files

`/classification`
> same as above but for classification tasks. Here, the bottom 90% of ecah dataset is considered toxic.

`multitask_evalutation_regression.ipynb`
> notebook loading results from hyperparameter optimization for all models in regression tasks, retraining them on the development data, and testing on the independant test sets.


`multitask_evalutation_classification.ipynb`
> same as above but for classification tasks
