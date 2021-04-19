import pandas as pd
import data_load_clean as dlc 
import rdkit
from rdkit import Chem
import rdkit.Chem.Descriptors
import deepchem as dc
import numpy as np
from sklearn import preprocessing
import sklearn.decomposition
import altair
import matplotlib.pyplot as plt
import classyfirepy as cp
from matplotlib import cm
import seaborn as sns


def load_data(csv_file: str, id_col: str):
    # need to change file path to where the data is on your local for now
    # later can decide what, if any, data we might want to put on github 
    df = pd.read_csv(csv_file)
    df_1 = df.drop_duplicates(subset=id_col)
    df_2 = df_1.reset_index()
    return df_2

def smiles_to_mol(dataframe, smiles_col):
    mols = []
    for smiles in dataframe[smiles_col]:
        mol =  Chem.MolFromSmiles(smiles)
        mols.append(mol)
    dataframe['Mol'] = mols
    return dataframe

def add_circular_fingerprint(df):
    featurizer = dc.feat.CircularFingerprint()
    f_list = [] 
    for mol in df['Mol']:
        f = featurizer.featurize(mol)
        f_list.append(f)
    df['CircularFingerprint'] = f_list
    return df

def preprocess(dataframe, drop_cols = ['smiles', 'target']):
    """
    Drops rows with missing values and converts categorical values to integers
    
    """
    
    # drops all rows with NaNs, although this can be modified to only look at rows in specific columns
    dataframe = dataframe.dropna(axis=0, subset = drop_cols) 
    
    # convert categorical values, if any exist, into integers
    for column in dataframe:
        if dataframe[column].dtype == str:
            l1 = preprocessing.LabelEncoder()
            l1.fit(dataframe[column])
            dataframe[column] = l1.transform(dataframe[column])
        else:
            continue

    return dataframe

def df_to_dataset(dataframe, X, y, w, ids):
    """
    Converts dataframe to DeepChem dataset object
    """
    
    to_dataset = dc.data.NumpyDataset(X, y, w, ids)
    dataset = to_dataset.from_dataframe(dataframe)
    
    return dataset

def analyze(dataframe, smiles_col, target_col, mol_col='Mol', classify=False):
    chart1, chart2, chart3, chart4, chart5, chart6 = [None,] * 6
    ## size of set
    print('Number of unique mols: ', len(dataframe))
    
    ## tox target
    y = dataframe[target_col].to_numpy()
    print('Target min, max: {}, {}'.format(y.min(), y.max()))
    
    ## hist of target
    chart1 = altair.Chart(dataframe[[target_col, smiles_col]]).mark_bar().encode(
    altair.X(target_col, bin=True),
    y='count()')
    
    ## molecular weight
    dataframe['mw'] = dataframe[mol_col].apply(lambda mol: rdkit.Chem.Descriptors.ExactMolWt(mol))
    
    ## hist mol weight
    chart2 = altair.Chart(dataframe[['mw', smiles_col]]).mark_bar().encode(
    altair.X("mw", bin=True),
    y='count()',
    )
    
    ## partial charge
    dataframe['max_abs_partial_charge'] = dataframe[mol_col].apply(lambda mol: rdkit.Chem.Descriptors.MaxAbsPartialCharge(mol))
    
    ## hist partial charge
    chart3 = altair.Chart(dataframe[['max_abs_partial_charge', smiles_col]]).mark_bar().encode(
    altair.X("max_abs_partial_charge", bin=True),
    y='count()',
    )
    
    ## classify
    if classify:
        dataframe, queries_classified = cp.dataframe_query_and_update(dataframe, smiles_col)
        dataframe = cp.expand_ClassyFied_df(dataframe)
        cdf = dataframe[['cf_superclass', 'cf_class', 'cf_subclass', smiles_col]].fillna('None')
        
        supergroups = cdf.groupby('cf_superclass')
        names = {'top':[], 'mid': [], 'bot':[]}
        counts = {'top':[], 'mid': [], 'bot':[]}

        for sx, (sg, sinds) in enumerate(supergroups.groups.items()):

            names['top'].append(sg)
            counts['top'].append(len(sinds))

            names['mid'].append([])
            counts['mid'].append([])

            names['bot'].append([])
            counts['bot'].append([])


            superclass = cdf[cdf['cf_superclass'] == sg]

            classgroups = superclass.groupby('cf_class')

            for cx, (cg, cinds) in enumerate(classgroups.groups.items()):

                names['mid'][sx].append(cg)
                counts['mid'][sx].append(len(cinds))

                names['bot'][sx].append([])
                counts['bot'][sx].append([])

                class_ = superclass[superclass['cf_class'] == cg]

                subclassgroups = class_.groupby('cf_subclass')

                for sux, (sug, suinds) in enumerate(subclassgroups.groups.items()):
                    names['bot'][sx][cx].append(sug)
                    counts['bot'][sx][cx].append(len(suinds))
                    
        counts_t = np.array(counts['top'])
        names_t = np.array(names['top'])
        rolling_ct_t = np.cumsum(counts_t)
        cst = cm.viridis(rolling_ct_t/counts_t.sum())

        counts_m = []
        names_m = []
        csm = []
        for i in range(len(counts['mid'])):
            cm_portion = np.tile(cst[i], (len(counts['mid'][i]),1))

            counts_m_portion = []
            for n in range(len(counts['mid'][i])):
                counts_m_portion.append(counts['mid'][i][n])
                names_m.append(names['mid'][i][n])

            counts_m.extend(counts_m_portion)
            cm_portion_weights = (np.cumsum(counts_m_portion)+20000)/\
                (np.sum(counts_m_portion)+20000)

            cm_portion[:,0:3] = cm_portion[:,0:3]*cm_portion_weights[:,None]
            csm.append(cm_portion)

        csm = np.concatenate(csm, axis=0)
        names_m = np.array(names_m)

        counts_b = []
        names_b = []
        csb = []
        x=-1
        for i in range(len(counts['bot'])):
            for n in range(len(counts['bot'][i])):
                x+=1
                cm_portion = np.tile(csm[x], (len(counts['bot'][i][n]),1))
                counts_b_portion = []
                for l in range(len(counts['bot'][i][n])):
                    counts_b_portion.append(counts['bot'][i][n][l])
                    names_b.append(names['bot'][i][n][l])

                counts_b.extend(counts_b_portion)
                cm_portion_weights = (np.cumsum(counts_b_portion)+1000)/\
                    (np.sum(counts_b_portion)+1000)

                cm_portion[:,0:3] = cm_portion[:,0:3]*cm_portion_weights[:,None]
                csb.append(cm_portion)

        csb = np.concatenate(csb, axis=0)
        names_b = np.array(names_b)
        
        fig, ax = plt.subplots()

        size = .5


        for cs, n in [(cst, names_t), (csm, names_m), (csb, names_b)]:
            mask = n == 'None'
            cs[mask] = np.zeros(np.shape(cs[mask]))

        patches, texts = ax.pie(counts_t, radius=3, colors=cst,
               wedgeprops=dict(width=size, edgecolor=(0,0,0,0)))
        plt.legend(patches, names_t, loc='center left', bbox_to_anchor=(-0.8, 1.2),
                   fontsize=14)

        ax.pie(counts_m, radius=3-size, colors=csm,
               wedgeprops=dict(width=size, edgecolor=(0,0,0,0)))

        ax.pie(counts_b, radius=3-2*size, colors=csb,
               wedgeprops=dict(width=size, edgecolor=(0,0,0,0)))

        ax.set(aspect="equal")
        plt.title('Classified', loc='center', fontdict={'size':35}, y=.4)
        plt.show()
        chart4 = fig
        
        subs_nested = dataframe['cf_substituents']
        substituents = []

        for slist in subs_nested:
            try:
                substituents.extend(slist)
            except:
                pass

        subs_df = pd.DataFrame({'Sub':substituents})
        
        sns.countplot(y='Sub',data = subs_df, orient='v', order=subs_df['Sub'].value_counts().iloc[:25].index)

        fig = plt.gcf()
        plt.ylabel('Substituent group')
        chart5 = fig
        
    ## PCA
    pca = sklearn.decomposition.PCA(2)
    
    X = np.stack(dataframe['CircularFingerprint'].values).reshape(len(dataframe), -1)
    X_pcs = pca.fit_transform(X)
    pc_dataframe = pd.DataFrame(data=X_pcs, columns = ['PC1', 'PC2'])
    
    chart6 = altair.Chart(pc_dataframe).transform_fold(
        ['PC1',
         'PC2'],
        as_ = ['Measurement_type', 'value']
    ).transform_density(
        density='value',
        bandwidth=0.3,
        groupby=['Measurement_type'],
        extent= [0, 5],
        counts = True,
        steps=200
    ).mark_area().encode(
        altair.X('value:Q'),
        altair.Y('density:Q', stack='zero'),
        altair.Color('Measurement_type:N')
    ).properties(width=400, height=100)
    
    return chart1, chart2, chart3, chart4, chart5, chart6