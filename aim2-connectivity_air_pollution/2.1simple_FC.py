import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter

from os.path import join

import math
import json
import matplotlib as mpl

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


font_path2 = '/Users/katherine.b/Library/Fonts/Raleway-Regular.ttf'  # Your font path goes here
fm.fontManager.addfont(font_path2)
regular = {'fontname':'Raleway'}

font_path = '/Users/katherine.b/Library/Fonts/Raleway-ExtraLight.ttf'  # Your font path goes here
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


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

stronger_by_network = pd.DataFrame()
weaker_by_network = pd.DataFrame()

for source in sources.keys():
    avg_corrs = corr_df.fillna(0).loc[source].mean()
    corrmat = pd.DataFrame(
        dtype=float,
        index=NETWORKS.values(),
        columns=NETWORKS.keys()
    )
    
    for varname in avg_corrs.index:
        network1 = varname.split('_')[3]
        network2 = varname.split('_')[5]
        if avg_corrs[varname] > 0.01:
            #print(avg_corrs[varname])
            #corrmat.at[NETWORKS[network1], network2] = avg_corrs[varname]
            #corrmat.at[NETWORKS[network2], network1] = avg_corrs[varname]
            corrmat.at[NETWORKS[network1], network2] = 1
            corrmat.at[NETWORKS[network2], network1] = 1
        else:
            corrmat.at[NETWORKS[network1], network2] = np.nan
            corrmat.at[NETWORKS[network2], network1] = np.nan
    temp = corrmat.sum(axis=1)
    temp.name = f'{source}B'
    stronger_by_network = pd.concat(
        [
            stronger_by_network,
            temp
        ],
        axis=1
    )
    triangle = pd.DataFrame(
        np.tril(corrmat),
        index=corrmat.index,
        columns=corrmat.columns
    )
    triangle = triangle.replace(
        {
            0: np.nan,
            #999: 0
        }
    )
    fig,ax = plt.subplots(figsize=(4,4))
    g = sns.heatmap(
        triangle, 
        cmap=cmaps[source], 
        #center=0, 
        vmax=0.04,
        vmin=-0.00,
        square=True, 
        ax=ax
    )
    ax.set_xticklabels([NTWKS[i] for i in NETWORKS.keys()])
    ax.set_yticklabels([NTWKS[i] for i in NETWORKS.keys()])
    ax.set_title(sources[source])
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'CPM-{source}-baseline_corrs+.png'),
        dpi=400,
        transparent=True,
        #bbox_inches='tight'
    )

    corrmat = pd.DataFrame(
        dtype=float,
        index=NETWORKS.values(),
        columns=NETWORKS.keys()
    )
    
    for varname in avg_corrs.index:
        network1 = varname.split('_')[3]
        network2 = varname.split('_')[5]
        #print(avg_corrs[varname])
        if avg_corrs[varname] < -0.01:
            #print('negative!')
            #print(avg_corrs[varname])
            #corrmat.at[NETWORKS[network1], network2] = -1 * avg_corrs[varname]
            #corrmat.at[NETWORKS[network2], network1] = -1 * avg_corrs[varname]
            corrmat.at[NETWORKS[network1], network2] = 1
            corrmat.at[NETWORKS[network2], network1] = 1
        else:
            corrmat.at[NETWORKS[network1], network2] = np.nan
            corrmat.at[NETWORKS[network2], network1] = np.nan
    temp = corrmat.sum(axis=1)
    temp.name = f'{source}B'
    weaker_by_network = pd.concat(
        [
            weaker_by_network,
            temp
        ],
        axis=1
    )
    triangle = pd.DataFrame(
        np.tril(corrmat),
        index=corrmat.index,
        columns=corrmat.columns
    )
    triangle = triangle.replace(
        {
            0: np.nan,
            #999: 0
        }
    )
    fig,ax = plt.subplots(figsize=(4,4))
    g = sns.heatmap(
        triangle, 
        cmap=cmaps[source], 
        #center=0, 
        vmax=0.04,
        vmin=0.00,
        square=True, 
        ax=ax
    )
    ax.set_xticklabels([NTWKS[i] for i in NETWORKS.keys()])
    ax.set_yticklabels([NTWKS[i] for i in NETWORKS.keys()])
    ax.set_title(sources[source])
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'CPM-{source}-baseline_corrs-.png'),
        dpi=400,
        transparent=True,
        #bbox_inches='tight'
    )

for source in sources.keys():
    avg_corrs = delta_corr_df.fillna(0).loc[source].mean()
    corrmat = pd.DataFrame(
        dtype=float,
        index=NETWORKS.values(),
        columns=NETWORKS.keys()
    )
    
    for varname in avg_corrs.index:
        network1 = varname.split('_')[3]
        network2 = varname.split('_')[5]
        if avg_corrs[varname] > 0.01:
            #print(avg_corrs[varname])
            #corrmat.at[NETWORKS[network1], network2] = avg_corrs[varname]
            #corrmat.at[NETWORKS[network2], network1] = avg_corrs[varname]
            corrmat.at[NETWORKS[network1], network2] = 1
            corrmat.at[NETWORKS[network2], network1] = 1
        else:
            corrmat.at[NETWORKS[network1], network2] = np.nan
            corrmat.at[NETWORKS[network2], network1] = np.nan
    temp = corrmat.sum(axis=1)
    temp.name = f'{source}Δ'
    stronger_by_network = pd.concat(
        [
            stronger_by_network,
            temp
        ],
        axis=1
    )
    triangle = pd.DataFrame(
        np.tril(corrmat),
        index=corrmat.index,
        columns=corrmat.columns
    )
    triangle = triangle.replace(
        {
            0: np.nan,
            #999: 0
        }
    )
    fig,ax = plt.subplots(figsize=(4,4))
    g = sns.heatmap(
        triangle, 
        cmap=cmaps[source], 
        #center=0, 
        vmax=0.04,
        vmin=-0.00,
        square=True, 
        ax=ax
    )
    ax.set_xticklabels([NTWKS[i] for i in NETWORKS.keys()])
    ax.set_yticklabels([NTWKS[i] for i in NETWORKS.keys()])
    ax.set_title(sources[source])
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'CPM-{source}-delta_corrs+.png'),
        dpi=400,
        transparent=True,
        #bbox_inches='tight'
    )

    corrmat = pd.DataFrame(
        dtype=float,
        index=NETWORKS.values(),
        columns=NETWORKS.keys()
    )
    
    for varname in avg_corrs.index:
        network1 = varname.split('_')[3]
        network2 = varname.split('_')[5]
        #print(avg_corrs[varname])
        if avg_corrs[varname] < -0.01:
            #print('negative!')
            #print(avg_corrs[varname])
            #corrmat.at[NETWORKS[network1], network2] = -1 * avg_corrs[varname]
            #corrmat.at[NETWORKS[network2], network1] = -1 * avg_corrs[varname]
            corrmat.at[NETWORKS[network1], network2] = 1
            corrmat.at[NETWORKS[network2], network1] = 1
        else:
            corrmat.at[NETWORKS[network1], network2] = np.nan
            corrmat.at[NETWORKS[network2], network1] = np.nan
    temp = corrmat.sum(axis=1)
    temp.name = f'{source}Δ'
    weaker_by_network = pd.concat(
        [
            weaker_by_network,
            temp
        ],
        axis=1
    )
    triangle = pd.DataFrame(
        np.tril(corrmat),
        index=corrmat.index,
        columns=corrmat.columns
    )
    triangle = triangle.replace(
        {
            0: np.nan,
            #999: 0
        }
    )
    fig,ax = plt.subplots(figsize=(4,4))
    g = sns.heatmap(
        triangle, 
        cmap=cmaps[source], 
        #center=0, 
        vmax=0.04,
        vmin=0.00,
        square=True, 
        ax=ax
    )
    ax.set_xticklabels([NTWKS[i] for i in NETWORKS.keys()])
    ax.set_yticklabels([NTWKS[i] for i in NETWORKS.keys()])
    ax.set_title(sources[source])
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'CPM-{source}-delta_corrs-.png'),
        dpi=400,
        transparent=True,
        #bbox_inches='tight'
    )
stronger_by_network.to_csv(
    join(PROJ_DIR, OUTP_DIR, 'CPM-stronger_by_network.csv')
)
weaker_by_network.to_csv(
    join(PROJ_DIR, OUTP_DIR, 'CPM-weaker_by_network.csv')
)