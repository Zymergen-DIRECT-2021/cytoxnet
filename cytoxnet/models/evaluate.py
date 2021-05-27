"""Evaluate baseline model and feature types on specified datasets"""
from typing import List, Union, Type

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import cytoxnet.dataprep.dataprep
import cytoxnet.dataprep.featurize
import cytoxnet.models.models
import cytoxnet.dataprep.io

# typing
DataFrame = Type[pd.core.frame.DataFrame]

def evaluate_crossval(datafile: Union[str, DataFrame],
                      ml_model: str,
                      feat_method: str,
                      target: Union[str, List[str]],
                      k: int = 5,
                      codex: str = None,
                      fit_kwargs: dict = {},
                      **kwargs):
    """Cross validated metrics for a set of model, dataset, and feature type.
    
    For a specified datafile, ML model, and featurizer, compute the k fold
    cross validated R2 and MSE.
    
    Parameters
    ----------
    datafile : str or DataFrame
        The file path, name in package, or dataframe to be used as the dev set.
    ml_model : str
        The name of the ML model to test.
    feat_method : str
        The name of the deepchem featurizer to use.
    target : str or list of str
        The column name in datafile of the target/s
    k : int
        The number of folds to use, default 5
    codex : str, default None
        The filepath to the compounds file containing smiles-feature pairings.
        If specified, featurization will attemp to extract already computed
        features from the codex.
    fit_kwargs : dict
        Keyword argument's to the model's fit method.
    **kwargs passed to the model constructor.
    
    Returns
    -------
    dict : The metrics mapped to dataset, featurizer, and model type.
    """
    if type(datafile) == str:
        dataframe = cytoxnet.dataprep.io.load_data(datafile)
    elif type(datafile) == pd.core.frame.DataFrame:
        dataframe = datafile.copy()
    else:
        raise TypeError(
            f'datafile should be str or dataframe not {type(datafile)}'
        )
    df_wfeat = cytoxnet.dataprep.featurize.add_features(
        dataframe,
        method=feat_method,
        codex=codex)
    # convert to dataset
    dataset = cytoxnet.dataprep.dataprep.convert_to_dataset(
        df_wfeat,
        X_col = feat_method,
        y_col = target
    )
    # transform data
    dataset, transformers = cytoxnet.dataprep.dataprep.data_transformation(
        dataset,
        transformations=['NormalizationTransformer'],
        to_transform=['y']
    )
    folds = cytoxnet.dataprep.dataprep.data_splitting(
        dataset, split_type='k', k=k
    )
    metrics = []
    for train, val in folds:
        tox_model = cytoxnet.models.models.ToxModel(
            ml_model, transformers=transformers, **kwargs
        )
        tox_model.fit(train, **fit_kwargs)
        metrics_ = tox_model.evaluate(
            val,
            metrics=['r2_score', 'mean_squared_error'],
            untransform=True)
        metrics.append(list(metrics_.values()))
    metrics = np.average(np.array(metrics), axis=0)
    out = {'datafile': datafile,
           'model': ml_model,
           'featurizer': feat_method,
           'r2': metrics[0],
           'MSE': metrics[1]}
    return out

def grid_evaluate_crossval(datafiles: List[Union[str, DataFrame]],
                           ml_models: List[str],
                           feat_methods: List[str],
                           targets_codex: dict,
                           k: int = 5,
                           parallel: bool = True,
                           **kwargs):
    """Cross validated metrics for a grid of models, datasets, and feats.
    
    For a specified grid of datafile, ML model, and featurizer, compute the
    k fold cross validated R2 and MSE, and return a dataframe of results.
    
    Parameters
    ----------
    datafiles : list of (str or DataFrame)
        The file paths, names in package, or dataframes to be
        used as the dev set.
    ml_models : list of str
        The names of the ML models to test.
    feat_methods : str
        The names of the deepchem featurizers to use.
    targets_codex : dict
        datafile: target column/s pairs. Needed to extract the right target/s
        from different datasets
    k : int
        The number of folds to use, default 5
    parallel : bool
        Whether to execute the grid in parallel.
    **kwargs passed to the evaluate_crossval function.
    """
    if parallel:
        output = Parallel(n_jobs=-1)(
            delayed(
                evaluate_crossval
            )(
                 datafile,
                 ml_model = model,
                 feat_method=feature,
                 target=targets_codex[datafile],
                 k=k,
                 **kwargs
            ) for datafile in datafiles for model in ml_models\
                  for feature in feat_methods
        )
        
    else:
        output = []
        for datafile in datafiles:
            for model in ml_models:
                for feature in feat_methods:
                    out = evaluate_crossval(
                        datafile,
                        ml_model = model,
                        feat_method=feature,
                        target=targets_codex[datafile],
                        k=k,
                        **kwargs
                    )
                    output.append(out)
    df = pd.DataFrame(output)
    return df