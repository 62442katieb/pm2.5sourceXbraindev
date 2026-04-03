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

def consistency_plots(df, name, thresh):
    stronger_by_network = pd.DataFrame()
    weaker_by_network = pd.DataFrame()
    consistent_conns = pd.DataFrame(
        0,
        columns=sources.values(),
        index=NTWKS.values()
    )
    for source in sources.keys():
        temp1 = df.loc[source].astype(float)
        avg_corrs = df.fillna(0).loc[source].mean()
        counts = temp1.describe().T.sort_values('count')['count']
        corrmat = pd.DataFrame(
            dtype=float,
            index=NTWKS.values(),
            columns=NTWKS.values()
        )
        
        for varname in counts.index:
            network1 = varname.split('_')[3]
            network2 = varname.split('_')[5]
            if counts[varname] >= thresh * len(df.index.get_level_values(1).unique()):
                
                consistent_conns.at[NTWKS[network1], sources[source]] += 1
                consistent_conns.at[NTWKS[network2], sources[source]] += 1
                if avg_corrs[varname] > 0:
                    #print(avg_corrs[varname])
                    #corrmat.at[NETWORKS[network1], network2] = avg_corrs[varname]
                    #corrmat.at[NETWORKS[network2], network1] = avg_corrs[varname]
                    corrmat.at[NTWKS[network1], network2] = 1
                    corrmat.at[NTWKS[network2], network1] = 1
                else:
                    corrmat.at[NTWKS[network1], network2] = np.nan
                    corrmat.at[NTWKS[network2], network1] = np.nan
            else:
                pass
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
        #ax.set_xticklabels([NTWKS[i] for i in NETWORKS.keys()])
        #ax.set_yticklabels([NTWKS[i] for i in NETWORKS.keys()])
        ax.set_title(sources[source])
        fig.savefig(
            join(PROJ_DIR, FIGS_DIR, f'CPM-{source}-{name}_corrs+{int(thresh * 100)}.png'),
            dpi=400,
            transparent=True,
            #bbox_inches='tight'
        )

        corrmat = pd.DataFrame(
            dtype=float,
            index=NTWKS.values(),
            columns=NTWKS.keys()
        )
        
        for varname in avg_corrs.index:
            network1 = varname.split('_')[3]
            network2 = varname.split('_')[5]
            #print(avg_corrs[varname])
            if counts[varname] >= thresh * len(df.index.get_level_values(1).unique()):
                #print(varname, counts[varname])
                if avg_corrs[varname] < 0:
                    #print(avg_corrs[varname])
                    #corrmat.at[NETWORKS[network1], network2] = avg_corrs[varname]
                    #corrmat.at[NETWORKS[network2], network1] = avg_corrs[varname]
                    corrmat.at[NTWKS[network1], network2] = 1
                    corrmat.at[NTWKS[network2], network1] = 1
                else:
                    corrmat.at[NTWKS[network1], network2] = np.nan
                    corrmat.at[NTWKS[network2], network1] = np.nan
            else:
                pass
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
        #ax.set_xticklabels([NTWKS[i] for i in NETWORKS.keys()])
        #ax.set_yticklabels([NTWKS[i] for i in NETWORKS.keys()])
        ax.set_title(sources[source])
        fig.savefig(
            join(PROJ_DIR, FIGS_DIR, f'CPM-{source}-{name}_corrs-{int(thresh * 100)}.png'),
            dpi=400,
            transparent=True,
            #bbox_inches='tight'
        )
    return consistent_conns

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
pm25 = sns.light_palette('#333333', as_cmap=True)

sns.set(context='paper', style='white', palette=ibm, font_scale=1.5)


#font_path2 = '/Users/katherine.b/Library/Fonts/Raleway-Regular.ttf'  # Your font path goes here
#fm.fontManager.addfont(font_path2)
#regular = {'fontname':'Raleway'}

#font_path = '/Users/katherine.b/Library/Fonts/Raleway-ExtraLight.ttf'  # Your font path goes here
#fm.fontManager.addfont(font_path)
#prop = fm.FontProperties(fname=font_path)

#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = prop.get_name()


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
    'F6': 'Industrial/Residual Fuel',
    'reshist_addr1_pm252016aa': 'PM2.5'
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
    'F6': industr,
    'reshist_addr1_pm252016aa': pm25

}

THRESH = 0.9

corr_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-corrs_pm25.pkl'))
sens_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-corrs_sensitivity.pkl'))
delta_corr_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-delta-corrs_pm25.pkl'))

dfs = {
    'base': corr_df,
    'sens': sens_df,
    'delta': delta_corr_df
}

for df in dfs:
    conns = consistency_plots(dfs[df], df, THRESH)
    conns.to_csv(join(PROJ_DIR, OUTP_DIR, f'consistent_networks_{df}.csv'))

