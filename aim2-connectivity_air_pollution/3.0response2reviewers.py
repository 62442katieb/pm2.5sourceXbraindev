

import numpy as np
import pandas as pd
import seaborn as sns


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

did_you_move = pd.read_csv(
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release5.1/abcd-data-release-5.1/core/mental-health/mh_p_le.csv",
    index_col=["src_subject_id", "eventname"],
    usecols=["src_subject_id", "eventname", "ple_move_past_yr_p"])

delta_ppts = pd.read_csv(
    join(PROJ_DIR, DATA_DIR, "data_qcd_delta-siemens.csv"),
    index_col=0
).index.unique()

base_ppts = pd.read_csv(
    join(PROJ_DIR, DATA_DIR, "data_qcd_base.csv"),
    index_col=0
).index.unique()

delta_move = did_you_move.loc[delta_ppts]
base_move = did_you_move.loc[base_ppts]
waves = ['baseline_year_1_arm_1', '1_year_follow_up_y_arm_1', '2_year_follow_up_y_arm_1']

for wave in waves:
    print('\t\t', wave)
    temp2 = delta_move.xs(wave, level=1)
    temp = base_move.xs(wave, level=1)
    print(
        f"baseline sample:\t{temp.sum()}\t{np.round(temp.sum()/ len(base_ppts) * 100, 2)}%", 
        "\n", 
        f"longitudinal sample: \t{temp2.sum()}\t{np.round(temp2.sum()/ len(delta_ppts) * 100, 2)}%")