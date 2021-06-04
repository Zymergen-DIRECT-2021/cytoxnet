import deepchem as dc
import numpy as np
import pandas as pd
import optuna
from functools import reduce

import cytoxnet.dataprep.io as io
import cytoxnet.dataprep.dataprep as dataprep
import cytoxnet.dataprep.featurize as feat
from cytoxnet.models.models import ToxModel
import cytoxnet.models.opt as opt

## !!!!!!temporary until database query works
fish = io.load_data('../database/fish.csv', cols=['smiles', 'fish_LC50'])
daphnia = io.load_data('../database/daphnia.csv', cols=['smiles', 'daphnia_EC50'])
algea = io.load_data('../database/algea.csv', cols=['smiles', 'algea_EC50'])
rat  = io.load_data('../database/rat.csv', cols=['smiles', 'rat_LD50'])
ecoli  = io.load_data('../database/ecoli.csv', cols=['smiles', 'ecoli_MIC'])

raw = reduce(
    lambda x, y: pd.merge(x, y, how='outer', on = 'smiles'),
    [fish, daphnia, algea, rat, ecoli]
)

data_f = feat.add_features(raw, method='RDKitDescriptors', codex='../database/compounds.csv')

# identifying the dev, test indexes
algea_only = data_f[~data_f.isna()['algea_EC50']]
algea_index = algea_only.index
test_index = algea_only.sample(frac=.2, random_state=0).index
baseline_index = algea_only.drop(index=test_index).index


# convert to dataset
baseline = dataprep.convert_to_dataset(
    data_f,
    X_col='RDKitDescriptors',
    y_col=[
        'algea_EC50'
    ]
).select(np.isin(data_f.index, algea_index))

# normalize it
baseline_normed, baseline_transformations = dataprep.data_transformation(
    baseline, transformations = ['NormalizationTransformer'],
    to_transform = ['y']
)
# split out dev and test
baseline_test = baseline_normed.select(np.isin(baseline_normed.ids, test_index))
baseline_dev = baseline_normed.select(np.isin(baseline_normed.ids, baseline_index))

# define search space for RFR model
baseline_r_search_space = {
    'n_estimators': (10, 300, 5), # int uniform from 10 to 300 by 5
    'criterion': ['mse', 'mae'], # choice
    'max_depth': [*range(5, 50, 5), None],
    'min_samples_split': (2, 10, 1),
    'min_samples_leaf': (1, 10, 1),
    'max_features': ['auto', 'sqrt', 'log2']
}

opt.hypopt_model(
    model_name = 'RFR',
    dev_set = baseline_dev,
    search_space = baseline_r_search_space,
    study_name = 'opt',
    study_db = "sqlite:///baseline_r.db",
    transformations = baseline_transformations,
    trials_per_cpu=10
)
