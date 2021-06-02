import optuna
import os
import sys

baseline_regression_study = optuna.create_study(
    study_name='opt',
    storage="sqlite:///baseline_r.db",
    direction='maximize',
    load_if_exists=True
)

impmean_regression_study = optuna.create_study(
    study_name='opt',
    storage="sqlite:///mean_r.db",
    direction='maximize',
    load_if_exists=True
)

interpute_regression_study = optuna.create_study(
    study_name='opt',
    storage="sqlite:///inter_r.db",
    direction='maximize',
    load_if_exists=True
)
graph_regr_study = optuna.create_study(
    study_name='opt',
    storage="sqlite:///graph_r.db",
    direction='maximize',
    load_if_exists=True
)
