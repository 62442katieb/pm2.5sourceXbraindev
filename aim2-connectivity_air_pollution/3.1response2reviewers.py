

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join

ibm = sns.color_palette(
    [
        '#648FFF',
        '#785EF0',
        '#DC267F',
        '#FE6100',
        '#FFB000',
        '#858686'
    ]
)

crustal = sns.light_palette('#648FFF', as_cmap=True)
sulfate = sns.light_palette('#785EF0', as_cmap=True)
biomass = sns.light_palette('#DC267F', as_cmap=True)
traffic = sns.light_palette('#FE6100', as_cmap=True)
nitrate = sns.light_palette('#FFB000', as_cmap=True)
industr = sns.light_palette('#858686', as_cmap=True)

sns.set(context='paper', style='white', palette=ibm, font_scale=1.5)


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim2"
LOCAL_DR = "/Users/katherine.b/Dropbox/Projects/scehsc_pilot/aim2-connectivity_air_pollution"
DATA_DIR = "data"
OUTP_DIR = "output"
FIGS_DIR = "figures"

sources = {
    'F1': 'Crustal Materials', # no exposure-related FC
    'F2': 'Ammonium Sulfates', # no exposure-related FC
    'F3': 'Biomass Burning',
    'F4': 'Traffic Emissions',
    'F5': 'Ammonium Nitrates',
    'F6': 'Industrial/Residual Fuel'
}


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

cmaps = {
    'F1': crustal,
    'F2': sulfate,
    'F3': biomass,
    'F4': traffic,
    'F5': nitrate,
    'F6': industr
}

corr_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-corrs.pkl'))
delta_corr_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-delta-corrs.pkl'))

# do all iterations show same directionality?
consistency = pd.DataFrame(
    index=corr_df.columns,
)
for source in sources.keys():
    empty = pd.DataFrame(
        index=NTWKS.keys(),
        columns=NTWKS.keys(),
        dtype=float
    )
    mask = np.triu(np.ones_like(empty, dtype=bool), k=1)
    print('\n', sources[source])
    temp = corr_df.loc[source].dropna(axis=1, how='all').astype(float)
    counts = temp.describe().T.sort_values('count')['count']
    minmax = temp.describe().T.sort_values('count')[['min', 'max', 'mean']]
    for conn in counts.index:
        if counts[conn] >= 15:
            ntwk1 = conn.split('_')[3]
            ntwk2 = conn.split('_')[5]
            empty.at[ntwk1, ntwk2] = minmax.loc[conn]['mean']
            empty.at[ntwk2, ntwk1] = minmax.loc[conn]['mean']
            if minmax.loc[conn]['min'] > 0:
                if minmax.loc[conn]['max'] > 0:
                    consistency.at[conn, f'{source}_base'] = 1
                elif minmax.loc[conn]['max'] < 0:
                    consistency.at[conn, f'{source}_base'] = 0
            elif minmax.loc[conn]['min'] < 0:
                if minmax.loc[conn]['max'] < 0:
                    consistency.at[conn, f'{source}_base'] = 1
                elif minmax.loc[conn]['max'] > 0:
                    consistency.at[conn, f'{source}_base'] = 0
    fig,ax = plt.subplots(figsize=(4,4))
    g = sns.heatmap(
        empty.fillna(0).loc[NTWKS.keys()][NTWKS.keys()], 
        square=True, cmap='RdBu_r', center=0, mask=mask
    )
    g.set_xticklabels(NTWKS.values())
    g.set_yticklabels(NETWORKS.values())
    g.set_title(sources[source])
    plt.show()
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'{source}_base_consistent75.png'),
        dpi=600,
        bbox_inches='tight'
    )
    empty.to_csv(
        join(PROJ_DIR, OUTP_DIR, f'{source}_base_consistent75.csv')
    )

# now for changes
for source in sources.keys():
    empty = pd.DataFrame(
        index=NTWKS.keys(),
        columns=NTWKS.keys(),
        dtype=float
    )
    mask = np.triu(np.ones_like(empty, dtype=bool), k=1)
    print('\n', sources[source])
    temp = delta_corr_df.loc[source].dropna(axis=1, how='all').astype(float)
    counts = temp.describe().T.sort_values('count')['count']
    minmax = temp.describe().T.sort_values('count')[['min', 'max', 'mean']]
    for conn in counts.index:
        if counts[conn] >= 10:
            ntwk1 = conn.split('_')[3]
            ntwk2 = conn.split('_')[5]
            empty.at[ntwk1, ntwk2] = minmax.loc[conn]['mean']
            empty.at[ntwk2, ntwk1] = minmax.loc[conn]['mean']
            if minmax.loc[conn]['min'] > 0:
                if minmax.loc[conn]['max'] > 0:
                    consistency.at[conn, f'{source}_delta'] = 1
                elif minmax.loc[conn]['max'] < 0:
                    consistency.at[conn, f'{source}_delta'] = 0
            elif minmax.loc[conn]['min'] < 0:
                if minmax.loc[conn]['max'] < 0:
                    consistency.at[conn, f'{source}_delta'] = 1
                elif minmax.loc[conn]['max'] > 0:
                    consistency.at[conn, f'{source}_delta'] = 0
    fig,ax = plt.subplots(figsize=(4,4))
    g = sns.heatmap(
        empty.fillna(0).loc[NTWKS.keys()][NTWKS.keys()], 
        square=True, cmap='RdBu_r', center=0, mask=mask
    )
    g.set_xticklabels(NTWKS.values())
    g.set_yticklabels(NETWORKS.values())
    g.set_title(sources[source])
    plt.show()
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'{source}_delta_consistent50.png'),
        dpi=600,
        bbox_inches='tight'
    )
    empty.to_csv(
        join(PROJ_DIR, OUTP_DIR, f'{source}_delta_consistent50.csv')
    )

consistency.dropna(
    how='all').to_csv(
        join(PROJ_DIR, OUTP_DIR, 'consistent_directionality.csv'))
