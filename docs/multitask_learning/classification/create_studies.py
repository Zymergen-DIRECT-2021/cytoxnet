import optuna
import os
import sys

baseline_class_study = optuna.create_study(
    study_name='opt',
    storage="sqlite:///baseline_c.db",
    direction='maximize',
    load_if_exists=True
)

impmean_class_study = optuna.create_study(
    study_name='opt',
    storage="sqlite:///mean_c.db",
    direction='maximize',
    load_if_exists=True
)

interpute_class_study = optuna.create_study(
    study_name='opt',
    storage="sqlite:///inter_c.db",
    direction='maximize',
    load_if_exists=True
)
graph_class_study = optuna.create_study(
    study_name='opt',
    storage="sqlite:///graph_c.db",
    direction='maximize',
    load_if_exists=True
)
