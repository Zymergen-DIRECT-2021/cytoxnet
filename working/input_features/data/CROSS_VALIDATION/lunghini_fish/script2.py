import analyze_feats as ef
import numpy as np
from joblib import Parallel, delayed
import pandas as pd

feature_list = ['CircularFingerprint', 
                'MACCSKeysFingerprint',
                'MordredDescriptors',
                #'PubChemFingerprint',
                #'CoulombMatrix',
                'RDKitDescriptors' 
                ]

model_list = ['LASSO', 'RFR', 'GPR']

output = Parallel(n_jobs=15)(delayed(ef.evaluate_crossval)('lunghini',
                                      ml_model = model,
                                      feat_method=feature,
                                      target='fish_LC50',
                                      save_fig=False) for model in model_list for feature in feature_list)
out = []
for i in output:
    tox_model, data_name, ml_model, feat_method, metrics = i
    tmp = [data_name, ml_model, feat_method, *metrics.values()]
    out.append(tmp)
    
out=np.array(out)
print(out)
out = pd.DataFrame(data=out, columns=['data_name', 'ml_model', 'feat_method', 'R2', 'MSE'])
out.to_csv('fish_data.csv')

"""
out = []
for i in output:
    tox_model, data_name, ml_model, feat_method, metrics = i 
    tmp = [data_name, ml_model, feat_method, *metrics.values()]
    out.append(tmp)
    print(i)

out =np.array(out)
np.savetxt('fish_data.dat', out, delimiter=',')
"""
"""
for model in model_list:
    for feature in feature_list:
        print(model, feature)
        if feature == 'CoulombMatrix':
            ef.evaluate('chembl_ecoli',
                        ml_model = model,
                        feat_method=feature,
                        target='MIC',
                        max_atoms=400,
                        save_fig=False)
        else:
            ef.evaluate('chembl_ecoli',
                         ml_model = model, 
                         feat_method=feature,
                         target='MIC',
                         save_fig=False)
"""
