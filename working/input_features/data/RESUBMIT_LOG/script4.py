import evaluate  as ef
import numpy as np
from joblib import Parallel, delayed
import pandas s pd

feature_list = ['CircularFingerprint', 
                'MACCSKeysFingerprint',
                'MordredDescriptors',
                #'PubChemFingerprint'
                #'CoulombMatrix',
                'RDKitDescriptors'
                ]

model_list = ['LASSO', 'RFR', 'GPR']

data_list = ['chembl_ecoli_MIC.csv', 'lunghini_algea_EC50.csv', 'lunghini_daphnia_EC50.csv', 'lunghini_fish_LC50.csv', 'zhu_rat_LD50.csv']


for dataset in datalist:


for dataset in data_list:
    print('DATASET IS ', dataset)
    if dataset == 'lunghini_algea_EC50.csv':
        output = Parallel(n_jobs=15)(delayed(ef.evaluate_crossval)('lunghini',
                                                                    ml_model = model,
                                                                    feat_method=feature,
                                                                    target='algea_EC50',
                                                                    save_fig=False,
                                                                    species='algea') for model in model_list for feature in feature_list)
    elif dataset == 'lunghini_fish_LC50.csv':
        output = Parallel(n_jobs=15)(delayed(ef.evaluate_crossval)('lunghini',
                                                                    ml_model = model,
                                                                    feat_method=feature,
                                                                    target='fish_LC50',
                                                                    save_fig=False) for model in model_list for feature in feature_list)
   
    elif dataset == 'lunghini_daphnia_EC50.csv':
        output = Parallel(n_jobs=15)(delayed(ef.evaluate_crossval)('lunghini',
                                                                   ml_model = model,
                                                                   feat_method=feature,
                                                                   target='daphnia_EC50',
                                                                   save_fig=False,
                                                                   species = 'daphnia') for model in model_list for feature in feature_list)

    elif dataset == 'chembl_ecoli_MIC.csv':
        output = Parallel(n_jobs=15)(delayed(ef.evaluate_crossval)('chembl_ecoli',
                                      ml_model = model,
                                      feat_method=feature,
                                      target='MIC',
                                      save_fig=False) for model in model_list for feature in feature_list)

    elif dataset == 'zhu_rat_LD50.csv':
        output = Parallel(n_jobs=15)(delayed(ef.evaluate_crossval)('zhu_rat',
                                      ml_model = model,
                                      feat_method=feature,
                                      target='LD50',
                                      save_fig=False) for model in model_list for feature in feature_list)

    out = []
    for i in output:
        tox_model, data_name, ml_model, feat_method, metrics = i
        tmp = [data_name, ml_model, feat_method, *metrics.values()]
        out.append(tmp)

    out=np.array(out)

    print(out)
    out = pd.DataFrame(data=out, columns=['data_name', 'ml_model', 'feat_method', 'R2', 'MSE'])
    out.to_csv(dataset + '.out.dat')



