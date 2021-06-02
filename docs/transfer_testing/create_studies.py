import optuna
import os
import sys

graph_regr_study = optuna.create_study(
    study_name='opt',
    storage="sqlite:///graph_r.db",
    direction='maximize',
    load_if_exists=True
)
