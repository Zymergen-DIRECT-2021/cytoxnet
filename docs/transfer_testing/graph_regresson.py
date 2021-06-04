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


rat  = io.load_data('../database/rat.csv', cols=['smiles', 'rat_LD50'])


data_f = feat.add_features(rat, method='ConvMolFeaturizer')

graph = data_f

graph_set = dataprep.convert_to_dataset(
    graph,
    X_col='ConvMolFeaturizer',
    y_col='rat_LD50'
)

graph_normed, graph_transformations = dataprep.data_transformation(
    graph_set, transformations = ['NormalizationTransformer'],
    to_transform = ['y']
)

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
    dev_set = graph_normed,
    search_space = graph_r_search_space,
    study_name = 'opt',
    study_db = "sqlite:///graph_r.db",
    transformations = graph_transformations,
    trials_per_cpu=15,
    eval_kwargs={'per_task_metrics': True},
    fit_kwargs={'nb_epoch': 25},
    model_kwargs={'mode': 'regression', 'tasks': ['rat_LD50']}
)