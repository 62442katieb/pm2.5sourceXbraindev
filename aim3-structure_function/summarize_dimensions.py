

import numpy as np
import pandas as pd
import seaborn as sns

rni_fmri_path = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/KBpilot_aim3_CCA/Baseline_analysis/PLSC/bootstrap_plsc_fmri(rni).csv'

rni_fmri = pd.read_csv(rni_fmri_path, index_col=0, header=0)

NETWORKS = {
    'vs': 'Visual',
    'ad': 'Auditory', 
    'smh': 'Somatosensory, Hand', 
    'smm': 'Somatosensory, Mouth',  
    'dla': 'Dorsal Attention',
    'vta': 'Ventral Attention',
    #'n', 
    'rspltp': 'Retrosplenial Temporal', 
    'ca': 'Cinguloparietal', 
    'cgc': 'Cingulo-Opercular', 
    'sa': 'Salience',
    'fo': 'Frontoparietal',
    'dt': 'Default Mode', 
}

NTWKS = {
    'vs': 'Vis',
    'ad': 'Aud', 
    'smh': 'SMH', 
    'smm': 'SMM',  
    'dla': 'DAttn',
    'vta': 'VAttn',
    #'n', 
    'rspltp': 'RsTp', 
    'ca': 'CPa', 
    'cgc': 'COpp', 
    'sa': 'SN',
    'fo': 'FPN',
    'dt': 'DMN', 
}

summary = pd.DataFrame(
    dtype=float,
    index=NETWORKS.keys(),
    columns=pd.MultiIndex.from_product(
        [rni_fmri.columns, ['positive', 'negative']]
    )
)

for ntwk in NETWORKS.keys():
    temp = rni_fmri.filter(like=ntwk, axis=0)
    temp = temp.mask(np.abs(temp) < 2.5)
    for dim in rni_fmri.columns:
        if np.any(temp[dim] > 0):
            sum_ = temp[temp[dim] > 0][dim].sum()
            summary.at[ntwk, (dim, 'positive')] = float(sum_)
        if np.any(temp[dim] < 0):
            sum_ = temp[temp[dim] < 0][dim].sum()
            summary.at[ntwk, (dim, 'negative')] = float(sum_)