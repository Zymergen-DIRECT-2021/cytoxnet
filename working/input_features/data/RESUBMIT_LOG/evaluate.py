import cytoxnet.models.models
import cytoxnet.dataprep.io
import cytoxnet.dataprep.featurize
import cytoxnet.dataprep.dataprep
import importlib
import rdkit
import pubchempy
import mordred
import numpy as np
import altair as alt
from altair_saver import save
alt.renderers.enable('altair_saver', fmts=['png'])

def evaluate(datafile, ml_model, feat_method, target, save_fig=True,  **kwargs):
    """"""
    # load desired data
    """
    data_loader = getattr(cytoxnet.dataprep.io, 'load_' + data_name)
    if 'species' in kwargs.keys():
        dataframe = data_loader(kwargs['species'])
        kwargs.pop('species')
    else:
        dataframe = data_loader()
    # convert mostring to mol 
    df_wmol = cytoxnet.dataprep.featurize.molstr_to_Mol(dataframe, 'smiles')
    # add column with desired feature type
    df_wfeat = cytoxnet.dataprep.featurize.add_features(df_wmol, method=feat_method, **kwargs) # **kwargs
    nans = df_wfeat[feat_method].apply(lambda x: np.isnan(x).any())
    df_wfeat = df_wfeat[~nans]
    """
    # NEW
    dataframe = cytoxnet.dataprep.io.load_data(datafile)
    df_wfeat = cytoxnet.dataprep.featurize.add_features(dataframe)


    # convert to dataset
    dataset = cytoxnet.dataprep.dataprep.convert_to_dataset(df_wfeat,
                                                        X_col = [feat_method],
                                                        y_col = target)
    # transform data
    dataset, transformers = cytoxnet.dataprep.dataprep.data_transformation(dataset,
                                                                           transformations=[ 'NormalizationTransformer'],
                                                                           to_transform=['y'])
    # split into train and test sets
    train_set, test_set = cytoxnet.dataprep.dataprep.data_splitting(dataset, split_type='train_test_split')

    np.savetxt('X.csv', train_set.X, delimiter=',')
    np.savetxt('y.csv', train_set.y, delimiter=',')
    # define and train model
    tox_model = cytoxnet.models.models.ToxModel(ml_model, transformers=transformers) # **kwargs for alpha test
    tox_model.fit(train_set) # **kwargs

    # evaluate
    metrics = tox_model.evaluate(test_set, metrics=['r2_score','mean_squared_error'], untransform=True)
    print('r2_score, MSE: ', metrics)

    # save figure
    if save_fig == True:
        chart = tox_model.visualize('pair_predict', test_set, untransform=True)
        save(chart, data_name + ml_model + feat_method + target + '.png')

    return (tox_model, data_name, ml_model, feat_method, metrics)


def evaluate_crossval(datafile, ml_model, feat_method, target, save_fig=True,  **kwargs):
    """"""
    dataframe = cytoxnet.dataprep.io.load_data(datafile)
    df_wfeat = cytoxnet.dataprep.featurize.add_features(dataframe, **kwargs) # **kwargs

    # convert to dataset
    dataset = cytoxnet.dataprep.dataprep.convert_to_dataset(df_wfeat,
                                                        X_col = [feat_method],
                                                        y_col = target)
    # transform data
    dataset, transformers = cytoxnet.dataprep.dataprep.data_transformation(dataset,
                                                                           transformations=[ 'NormalizationTransformer'],
                                                                           to_transform=['y'])
    folds = cytoxnet.dataprep.dataprep.data_splitting(dataset, split_type='k', k=5)
    metrics = []
    for train, val in folds:
        tox_model = cytoxnet.models.models.ToxModel(ml_model, transformers=transformers)
        tox_model.fit(train)
        metrics_ = tox_model.evaluate(val, metrics=['r2_score', 'mean_squared_error'], untransform=True)
        metrics.append([metrics_['metric-1'], metrics_['metric-2']])
    metrics = np.average(np.array(metrics), axis=0)
    metrics = {'r2': metrics[0], 'MSE': metrics[1]}

    return (tox_model, datafile, ml_model, feat_method, metrics)


def evaluate_descriptors(dataname, descriptors_name, target, **kwargs):
    """"""
    trained_model = evaluate(data_name=dataname,
                             ml_model='LASSO',
                             feat_method=descriptors_name,
                             target=target,
                             save_fig=False,
                             **kwargs)[0]
    print(repr(trained_model.model.model.coef_))
    return


