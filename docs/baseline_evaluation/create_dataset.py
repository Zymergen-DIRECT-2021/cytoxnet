"""Create the database codex to speed up the search."""

import cytoxnet.dataprep.io as io

io.create_compound_codex()

datasets = ['lunghini_fish_LC50',
            'lunghini_daphnia_EC50',
            'lunghini_algea_EC50',
            'zhu_rat_LD50',
            'chembl_ecoli_MIC']
names = ['fish', 'daphnia', 'algea', 'rat', 'ecoli']
featurizers = ['CircularFingerprint',
               'RDKitDescriptors',
               'MACCSKeysFingerprint']

io.add_datasets(datasets, names=names, new_featurizers=featurizers)
