import cytoxnet.models.evaluate as ev

targets_codex = {
    'lunghini_fish_LC50': 'fish_LC50',
    'lunghini_daphnia_EC50': 'daphnia_EC50',
    'lunghini_algea_EC50': 'algea_EC50',
    'zhu_rat_LD50': 'rat_LD50',
    'chembl_ecoli_MIC': 'ecoli_MIC'
}
datafiles = [
    'lunghini_fish_LC50',
    'lunghini_daphnia_EC50',
    'lunghini_algea_EC50',
    'zhu_rat_LD50',
    'chembl_ecoli_MIC'
]
ml_models = ['RFR', 'GPR', 'LASSO']
featurizers = [
    'MordredDescriptors',
    'CircularFingerprint',
    'RDKitDescriptors',
    'MACCSKeysFingerprint'
]

featurizers.reverse()
datafiles.reverse()

df = ev.grid_evaluate_crossval(
    datafiles,
    ml_models,
    featurizers,
    targets_codex,
    parallel=False,
    codex='database/compounds.csv'
)
df.to_csv('regression_grid_results2.csv')
