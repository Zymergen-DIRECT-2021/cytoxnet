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

multitask_names = [
    'fish_LC50',
    'daphnia_EC50',
    'algea_EC50',
    'rat_LD50',
    'ecoli_MIC'
]

data_f = feat.add_features(raw, method='ConvMolFeaturizer')

# identifying the dev, test indexes
algea_only = data_f[~data_f.isna()['algea_EC50']]
algea_index = algea_only.index
test_index = algea_only.sample(frac=.2, random_state=0).index
baseline_index = algea_only.drop(index=test_index).index

graph = data_f.copy()

graph_set = dataprep.convert_to_dataset(
    graph,
    X_col='ConvMolFeaturizer',
    y_col=multitask_names
)
graph_set = dataprep.handle_sparsity(graph_set)

graph_normed, graph_transformations = dataprep.data_transformation(
    graph_set, transformations = ['NormalizationTransformer'],
    to_transform = ['y']
)

graph_test = graph_normed.select(np.isin(graph_normed.ids, test_index))
graph_dev = graph_normed.select(~np.isin(graph_normed.ids, test_index))

# define search space for RFR model
graph_r_search_space = {
    'dense_layer_size': (64, 320, 12),
    'dropout': (0.0, 0.5),
    'number_atom_features': (25, 125, 25),
    'batch_size': (50, 300, 25),
    'graph_conv_layers': [
        [32,],
        [64,],
        [128,],
        [32,32],
        [64,64],
        [128,128],
        [32,32,32],
        [64,64,64],
        [128,128,128]
    ]
}

opt.hypopt_model(
    model_name = 'GraphCNN',
    dev_set = graph_dev,
    search_space = graph_r_search_space,
    study_name = 'opt',
    study_db = "sqlite:///graph_r.db",
    transformations = graph_transformations,
    trials_per_cpu=15,
    target_index=2,
    eval_kwargs={'per_task_metrics': True, 'use_sample_weights':True},
    fit_kwargs={'nb_epoch': 25},
    model_kwargs={'mode': 'regression', 'tasks': multitask_names}
)