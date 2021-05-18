import cytoxnet.models.models
import cytoxnet.dataprep.io
import cytoxnet.dataprep.featurize
import cytoxnet.dataprep.dataprep
import importlib
import rdkit
import pubchempy
import mordred
import altair as alt
from altair_saver import save
alt.renderers.enable('altair_saver', fmts=['png'])

def evaluate(data_name, ml_model, feat_method, target, save_fig=True,  **kwargs):
    """"""
    # load desired data
    data_loader = getattr(cytoxnet.dataprep.io, 'load_' + data_name)
    dataframe = data_loader()
    # convert mostring to mol 
    df_wmol = cytoxnet.dataprep.featurize.molstr_to_Mol(dataframe, 'smiles')
    # add column with desired feature type
    df_wfeat = cytoxnet.dataprep.featurize.add_features(df_wmol, method=feat_method) # **kwargs
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
    
    # define and train model
    tox_model = cytoxnet.models.models.ToxModel(ml_model, transformers=transformers, **kwargs)
    tox_model.fit(train_set) # **kwargs

    # evaluate
    metrics = tox_model.evaluate(test_set, metrics=['r2_score','mean_squared_error'], untransform=True)
    print('r2_score, MSE: ', metrics)
    
    # save figure
    if save_fig == True:
        chart = tox_model.visualize('pair_predict', test_set, untransform=True)
        save(chart, data_name + ml_model + feat_method + target + '.png')
    
    return tox_model
