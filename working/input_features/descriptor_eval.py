"""
import cytoxnet.models.models
import cytoxnet.dataprep.io
import cytoxnet.dataprep.featurize
import cytoxnet.dataprep.dataprep
import importlib
import rdkit
import pubchempy
import mordred
"""
import eval_feat as ef

def evaluate_descriptors(dataname, descriptors_name, target, **kwargs):
    """"""
    trained_model = ef.evaluate(data_name=dataname,
                                ml_model='LASSO',
                                feat_method=descriptors_name,
                                target=target,
                                save_fig=False,
                                **kwargs)
    print(repr(trained_model.model.model.coef_))
    return 
