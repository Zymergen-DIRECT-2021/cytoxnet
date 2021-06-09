### Model evaluation results
Goal: Improve upon single target training of a small target set, in this case Lunghini algea data.
All models reported scores on independant algea EC50 test set not seen during h.opt or final training.
24hr+ multicore hyperparameter optimization using Tree-structured Parzen h.param sampling
conducted for each model reported.

Sparse weights applied both to training and evaluation for graph models.

### Hyperparameter optimization

Regression

| method | validation R2 score | best parameters |
| ------ | ------------------- | --------------- |
| baseline RFR single task | 0.523 | {'criterion': 'mse', 'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 295} |
| mean imputed RFR multitask | 0.121 | {'criterion': 'mse', 'max_depth': 35, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 290} |
| interpolated RFR multitask | 0.050 | {'criterion': 'mse', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 6, 'min_samples_split': 5, 'n_estimators': 195} |
| graph CNN multitask with sparse weights | 0.442 | {'batch_size': 275, 'dense_layer_size': 88, 'dropout': 0.05190727030105664, 'graph_conv_layers': \[128\], 'number_atom_features': 50} |
| baseline graph CNN single task | 0.536 | {'batch_size': 50, 'dense_layer_size': 76, 'dropout': 0.0024988702928001455, 'graph_conv_layers': \[128, 128, 128\], 'number_atom_features': 25} |

Classification

| method | validation precision score | best parameters |
| ------ | -------------------------- | --------------- |
| baseline RFC single task | 0.924 | {'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 10} |
| mean imputed RFC multitask | 0.960 | {'criterion': 'entropy', 'max_depth': 25, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 10} |
| interpolated RFC multitask | 0.916 | {'criterion': 'gini', 'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 10} |
| graph CNN multitask with sparse weights | -0.108 | {'batch_size': 275, 'dense_layer_size': 112, 'dropout': 0.013039897074701816, 'graph_conv_layers': \[64\], 'number_atom_features': 25} |

### Multitask model evaluation: regression
Reported R2 score

| method | R2 Score |
| ------ | -------- |
| baseline RFR single task | 0.475 |
| mean imputed RFR multitask | 0.174 |
| interpolated RFR multitask | 0.135 |
| graph CNN multitask with sparse weights | 0.405 |

Multitask RFR does not seem to pose an advantage with the imputation strategies tests. Multitask GCNN with sparse targets weighted 0 recovers most of the accuracy, but is still worse than the baseling.

### Multitask model evaluation: classification
Reported precision score.
Toxicity metrics in the top 90% of values were labeled toxic, and the remaining labeled nontoxic. This was done due to the imbalance of the datasets towards toxic compounds. Optimizing with respect to precision score is thus finding the model most capable of not labeling nontoxic compounds as toxic.

| method | Precision Score |
| ------ | -------- |
| baseline RFC single task | 0.906 |
| mean imputed RFC multitask | 0.455 |
| interpolated RFC multitask | 0.477 |
| graph CNN multitask with sparse weights | 0.919 |

Imputation is detrimental in the classification case: imputed values are all toxic (in the case of mean) and mostly toxic (in the case of interpolated) thus increaseing the dataset bias towards toxic even further. Multitask GCNNs with sparse weights actually improve on the random forest single task baseline, indicating that it is possible to leverage larger extra-sepcies datasets to improve accuracy.

### Transfer learning evaluation: regression
Reported R2 score. Hyperparameter optimzation covers only initial training on large dataset, thus it is the optimum model parameters associated with a raw single task GCNN. The baseline is trained on a development set of algea data, while the transfer model is first trained for longer on the much larger Zhu Rat LD50 dataset before training on the algea data with the graph convolution layers fixed as untrainable, leaving only the dense regressor of the model to retrain to algea data. 

| method | R2 |
| ------ | -------- |
| baseline GCNN | 0.380 |
| GCNN transfered from large dataset | 0.399 |

Training on the larger rat dataset, fixing the layers, and then retraining only the output regressor on the small algea dataset offered modest improvement of the test set compared to training on the small algea set alone.
