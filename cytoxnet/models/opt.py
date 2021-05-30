from typing import Type, List, Union

import deepchem as dc
import optuna
import numpy as np

import cytoxnet.models.models as md
import cytoxnet.dataprep.dataprep as dataprep

Dataset = Type[dc.data.NumpyDataset]

def hypopt_model(
    model_name: str,
    dev_set: Dataset,
    search_space: dict,
    study_name: str,
    target_index: int = None,
    study_db: str = "sqlite:///optimization.db",
    transformations: list = [],
    metric: list = 'r2_score',
    cv: int = 5,
    trials_per_cpu: int = 10,
    model_kwargs: dict = {},
    fit_kwargs: dict = {},
    eval_kwargs: dict = {}
):
    # check no overlap between fixed hparams and search space
    assert not np.any(np.isin(
        np.array(search_space.keys()),
        np.array(model_kwargs.keys())
    )), 'Cannot have fixed hyperparameters and searched hyperparameters'
    
    # split the data according to cv
    splits = dataprep.data_splitting(dev_set, split_type='k', k=cv)
    
    # prepare the search space by pairing with the correct optuna function
    searchable_space = []
    for name, args in search_space.items():
        searchable_space.append(SearchableSpace(name, args))
    
    # create objective function
    # ###########################################################
    def objective(trial):
        # handle the searchable space
        search_kwargs = {}
        for space in searchable_space:
            search_kwargs[space.name] = space.func(
                trial, space.name, *space.args
            )
        
        # run through the cv
        fold_results = []
        for train, val in splits:
            model = md.ToxModel(
                model_name,
                transformers=transformations,
                **search_kwargs,
                **model_kwargs)
            model.fit(train, **fit_kwargs)
            metric_ = list(
                model.evaluate(val, [metric], untransform=True, **eval_kwargs).values()
            )[0]
            if hasattr(metric_, '__len__'):
                assert target_index != None, 'Metrics for multiple targets\
 returned, pass target_index to choose the one desired for optimization.'
                metric_ = metric_[target_index]
            fold_results.append(metric_)
        metric_result = np.average(fold_results)
        return metric_result
    # ##########################################################
    # load the study and execute it
    study = optuna.load_study(
        study_name=study_name, storage=study_db
    )
    study._storage = study._storage._backend
    study.optimize(objective, n_trials=trials_per_cpu)
    return 

# E. Komp's code from Gandy Project Wi 2021
class SearchableSpace:
    """Wrapper to convert user specified search space into Optuna readable
    function.
    Args:
        hypname (str):
            Name of hyperparameter.
        space (tuple or list):
            The user defined hyperparameter space to search, determined by form
            Options -
                list -> options to choose from uniformly
                tuple of int -> sample integers in
                    (low, high, step[optional, d=1])
                tuple of float -> sample floats in
                    (low, high, distribution[optional, d='uniform'])
                    distribution options :
                        'uniform' -> uniform continuous
                        'loguniform' -> loguniform continuous
                        float -> uniform discrete
    Attributes:
        func (optuna.trials.Trial method):
            Function to be used for sampling of hyperparams.
        hypname (str):
            Name of hyperparameter.
        args (tuple): Positional arguments after name to be passed to func for
            sampling
    """

    def __init__(self, hypname, space):
        # pseudocode
        # . if statement format of space
        #      set self.func, self.args, and self.hypname
        self.name = hypname

        # categorical
        if isinstance(space, list):
            self.args = (space,)
            self.func = optuna.trial.Trial.suggest_categorical
        # others
        elif isinstance(space, tuple):
            # check if we need to add a parameter to the end (len =2)
            if len(space) == 2:
                if all(isinstance(i, int) for i in space):
                    space_ = list(space)
                    space_.append(1)
                    space_ = tuple(space_)
                    print(
                        'Assuming uniform integer sampling for hyperparameter\
 {} with search space specified as Tuple[int] with len 2'.format(hypname)
                    )
                elif all(isinstance(i, float) for i in space):
                    space_ = list(space)
                    space_.append('uniform')
                    space_ = tuple(space_)
                    print(
                        'Assuming uniform continuous sampling for\
 hyperparameter {} with search space specified as Tuple[float] with\
  len 2'.format(hypname)
                    )
                else:
                    raise ValueError('hyperparameter space as tuple must have\
 the first two arguments be both float or integer')

            elif len(space) == 3:
                space_ = space
            else:
                raise ValueError(
                    'space as a tuple indicates (min, max, step/type) and\
 should have 2 or 3 contents, not {}'.format(len(space)))

            if not isinstance(space_[0], type(space_[1])):
                raise ValueError('hyperparameter space as tuple must have\
 the first two arguments be both float or integer')
            # integer choice
            elif isinstance(space_[0], int):
                if not isinstance(space_[2], int):
                    raise ValueError('First two values in space are int,\
 indicating integer selection, but the third (step size) is not an int')
                else:
                    pass
                self.args = space_
                self.func = optuna.trial.Trial.suggest_int
            elif isinstance(space_[0], float):
                if space_[2] == 'uniform':
                    self.args = space_[:2]
                    self.func = optuna.trial.Trial.suggest_uniform
                elif space_[2] == 'loguniform':
                    self.args = space_[:2]
                    self.func = optuna.trial.Trial.suggest_loguniform
                elif isinstance(space_[2], float):
                    self.args = space_
                    self.func = optuna.trial.Trial.suggest_discrete_uniform
                else:
                    raise ValueError(
                        'Unknown specification for float suggestion {}, should\
 be "uniform" or "loguniform" indicating the distribution, or a float,\
  indicating a discrete spep'
                    )

            else:
                raise ValueError('hyperparameter space as tuple must have\
 the first two arguments be both float or integer')

        else:
            raise TypeError(
                'space must be a list or tuple, not {}'.format(type(space))
            )
        return