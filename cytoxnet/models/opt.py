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
    metric: str = 'r2_score',
    cv: int = 5,
    trials_per_cpu: int = 10,
    model_kwargs: dict = {},
    fit_kwargs: dict = {},
    eval_kwargs: dict = {}
):
    """
    Optimize a specified ToxModel by name over hyperperameter space.
    
    For a ToxModel and a development dataset, search for the best
    hyperparameter set over a specified search window. Optimizing for
    a specified metric. Uses cross validation. Can be run multiple times,
    on multiple cpus by simply executing the function again on each worker.
    `mpirun` is a quick solution to scatter to many workers.
    
    Parameters
    ----------
    model_name : str
        Name of ToxModel type under investigation
    dev_set : deepchem.data.NumpyDataset
        Dataset used for searching for the best parameters.
    search_space : dict of hyperparameter name: search space
        The form of values in the dict determines how the hyperparameter is
        sampled.
        Options -
            list -> options to choose from uniformly
            tuple of int -> sample integers in
                (low, high, step[optional, d=1])
            tuple of float -> sample floats in
                (low, high, distribution[optional, d='uniform'])
                distribution options :
                    'uniform' -> uniform continuous
                    'loguniform' -> loguniform continuous
                    float -> uniform discrete\
    study_name : str
        The name of the study stored in the study database to commit trials to.
    target_index : int
        If the target is multindex, and per_task_metric is passed to evaluate
        keywords, must be specified to determine which target is to be
        optimized to.
    study_db : str
        The storage database containing the study to save optimization to.
    transformations : list of :obj:deepchem.transformations
        The transformations applied to the data to be reveresed for eval
    metric : str
        Metric name in deepchem to use for evaluation
    cv : int
        Number of folds to conduct cross validation for.
    trials_per_cpu : int
        Number of trials to run whenever the function is executed, on each cpu.
    model_kwargs : dict
        Keyword arguments passed to model init NOT searched over in
        search_space
    fit_kwargs : dict
        Keyword arguments passed to the fit method
    eval_kwargs : dict
        Keyword arguments passed to the evaluate method.
    """
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
            metric_ = model.evaluate(val, [metric], untransform=True, **eval_kwargs)
            if type(metric_) == tuple:
                print('Tuple: ', metric_)
                if target_index is None:
                    print('Multiple targets found, using average score.')
                    metric_ = list(metric_[0].values())[0]
                else:
                    metric_ = list(metric_[1].values())[0][target_index]
            elif type(metric_) == dict:
                metric_ = list(metric_.values())[0]
            fold_results.append(metric_)
        metric_result = np.average(fold_results)
        return metric_result
    # ##########################################################
    # load the study and execute it
    study = optuna.load_study(
        study_name=study_name, storage=study_db
    )
    study._storage = study._storage._backend
    
    trials_counted = 0
    while trials_counted < trials_per_cpu:
        try:
            study.optimize(objective, n_trials=1)
            trials_counted +=1
        except:
            raise
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