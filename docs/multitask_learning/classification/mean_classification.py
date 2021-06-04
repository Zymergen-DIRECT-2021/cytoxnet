import deepchem as dc
import numpy as np
import pandas as pd
import optuna
from functools import reduce
import sklearn.impute

import cytoxnet.dataprep.io as io
import cytoxnet.dataprep.dataprep as dataprep
import cytoxnet.dataprep.featurize as feat
from cytoxnet.models.models import ToxModel
import cytoxnet.models.opt as opt

## !!!!!!temporary until database query works
fish = io.load_data('../../database/fish.csv', cols=['smiles', 'fish_LC50'])
daphnia = io.load_data('../../database/daphnia.csv', cols=['smiles', 'daphnia_EC50'])
algea = io.load_data('../../database/algea.csv', cols=['smiles', 'algea_EC50'])
rat  = io.load_data('../../database/rat.csv', cols=['smiles', 'rat_LD50'])
ecoli  = io.load_data('../../database/ecoli.csv', cols=['smiles', 'ecoli_MIC'])

raw = reduce(
    lambda x, y: pd.merge(x, y, how='outer', on = 'smiles'),
    [fish, daphnia, algea, rat, ecoli]
)

multitask_names = [
    'fish_LC50',
    'daphnia_EC50',
    'algea_EC50',
    'rat_LD50',
    'ecoli_MIC'
]

data_f = feat.add_features(raw, method='RDKitDescriptors', codex='../../database/compounds.csv')

# identifying the dev, test indexes
algea_only = data_f[~data_f.isna()['algea_EC50']]
algea_index = algea_only.index
test_index = algea_only.sample(frac=.2, random_state=0).index
baseline_index = algea_only.drop(index=test_index).index

mean = data_f.copy()
mean[multitask_names] = sklearn.impute.SimpleImputer().fit_transform(
    mean[multitask_names].values
)

# binarize target
mean = dataprep.binarize_targets(mean, target_cols=multitask_names, percentile = .9)

mean_set = dataprep.convert_to_dataset(
    mean,
    X_col='RDKitDescriptors',
    y_col=multitask_names
)

mean_test = mean_set.select(np.isin(mean_set.ids, test_index))
mean_dev = mean_set.select(~np.isin(mean_set.ids, test_index))

# define search space for RFR model
mean_c_search_space = {
    'n_estimators': (10, 300, 5), # int uniform from 10 to 300 by 5
    'criterion': ['gini', 'entropy'], # choice
    'max_depth': [*range(5, 50, 5), None],
    'min_samples_split': (2, 10, 1),
    'min_samples_leaf': (1, 10, 1),
    'max_features': ['auto', 'sqrt', 'log2']
}

opt.hypopt_model(
    model_name = 'RFC',
    dev_set = mean_dev,
    search_space = mean_c_search_space,
    study_name = 'opt',
    study_db = "sqlite:///mean_c.db",
    trials_per_cpu=10,
    target_index=2,
    metric='precision_score',
    eval_kwargs={'n_classes': 2, 'per_task_metrics': True}
)