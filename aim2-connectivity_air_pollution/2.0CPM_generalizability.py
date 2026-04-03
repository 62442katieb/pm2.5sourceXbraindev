import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
#import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter

from os.path import join
from scipy.stats import spearmanr
#from scipy.spatial.distance import dice, jaccard
#from nilearn import plotting, surface, datasets
#from geopy.geocoders import Nominatim
import geopandas as gpd
import geodatasets

import math
import json
import matplotlib as mpl

def consistency(df, thresh, return_adj=False):
    consistency = pd.Series(
        index=df.columns,
    )
    num_sites = len(df.index) * (thresh / 100)
    adj = pd.DataFrame(
        index=NETWORKS,
        columns=NETWORKS,
        dtype=float
    )
    mask = np.triu(np.ones_like(adj, dtype=bool), k=1)
    print('\n', sources[source])
    temp = df.astype(float)
    counts = temp.describe().T.sort_values('count')[['count', 'mean']]
    for conn in counts.index:
        if counts.loc[conn]['count'] >= num_sites:
            ntwk1 = conn.split('_')[3]
            ntwk2 = conn.split('_')[5]
            adj.at[ntwk1, ntwk2] = counts.loc[conn]['mean']
            adj.at[ntwk2, ntwk1] = counts.loc[conn]['mean']
            consistency.at[conn] = counts.loc[conn]['mean']
    if return_adj:
        return consistency.dropna(), adj
    else:
        return consistency.dropna()
    
#gdf = gpd.read_file('/Users/katherine.b/Dropbox/Mac/Downloads/cb_2018_us_state_500k(1)')
#gdf.head()

usa = gpd.read_file(geodatasets.get_path('geoda.natregimes'))

sns.set(context='paper', style='white', palette='husl', font_scale=1.5)


#font_path2 = '/Users/katherine.b/Library/Fonts/Raleway-Regular.ttf'  # Your font path goes here
#fm.fontManager.addfont(font_path2)
#regular = {'fontname':'Raleway'}

#font_path = '/Users/katherine.b/Library/Fonts/Raleway-ExtraLight.ttf'  # Your font path goes here
#fm.fontManager.addfont(font_path)
#prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = "Helvetica"


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim2"
LOCAL_DR = "/Users/katherine.b/Dropbox/Projects/scehsc_pilot/aim2-connectivity_air_pollution"
DATA_DIR = "data"
OUTP_DIR = "output"
FIGS_DIR = "figures"

index_to_site = {
    0: 'site20',
    1: 'site09',
    2: 'site13',
    3: 'site02',
    4: 'site10',
    5: 'site01',
    6: 'site04',
    7: 'site06',
    8: 'site16',
    9: 'site14',
    10: 'site05',
    11: 'site17',
    12: 'site21',
    13: 'site08',
    14: 'site03',
    15: 'site18',
    16: 'site07',
    17: 'site11',
    18: 'site15',
    19: 'site12',
    20: 'site19'
}

site_names = {'site01': "CHLA", 'site02': "CUB", 'site03': "FIU", 'site04': "LIBR",
              'site05': "MUSC", 'site06': "OHSU", 'site07': "ROC", 'site08': "SRI",
              'site09': "UCLA", 'site10': "UCSD", 'site11': "UFL", 'site12': "UMB",
              'site13': "UMICH", 'site14': "UMN", 
              #'site15': "UPMC", 
              'site16': "UTAH",
              'site17': "UVM", 
              'site18': "UWM",  'site19': "VCU", 'site20': "WUSTL",
              'site21': "YALE"}

site_cities = {
    'site01': "Los Angeles, CA", 
    'site02': "Boulder, CO", 
    'site03': "Miami, FL", 
    'site04': "Tulsa, OK",
    'site05': "Charleston, SC", 
    'site06': "Portland, OR", 
    'site07': "Rochester, NY", 
    'site08': "Menlo Park, CA",
    'site09': "Beverly Hills, CA", 
    'site10': "San Diego, CA", 
    'site11': "Gainesville, FL", 
    'site12': "Baltimore, MD",
    'site13': "Ann Arbor, MI",
    'site14': "Minneapolis, MN", 
    #'site15': "Pittsburgh, PA", 
    'site16': "Salt Lake City, UT",
    'site17': "Burlington, VT", 
    'site18': "Madison, WI",  
    'site19': "Richmond, VA", 
    'site20': "St. Louis, MO",
    'site21': "New Haven, CT"
}

regions = {'west': [
    'OHSU',
    'CUB',
    'SRI',
    'UCSD',
    'CHLA',
    'UCLA',
    'UTAH'
],

'southwest': ['LIBR'],

'southeast': [
    'MUSC',
    'UFL',
    'FIU',
    'VCU'
],

'midwest': [
    'UMICH',
    'UMN',
    'UWM',
    'WUSTL'
],

'northeast': [
    'ROC',
    'YALE',
    'UVM',
    'UMB',
    'UPMC'
]}

geo_order = {
    'site01': "CHLA", 
    'site02': "CUB", 
    'site16': "UTAH",
    'site08': "SRI",
    'site09': "UCLA", 
    'site10': "UCSD", 
    'site06': "OHSU", 
    'site04': "LIBR",
    'site13': "UMICH", 
    'site14': "UMN", 
    'site18': "UWM",  
    'site20': "WUSTL",
    'site07': "ROC", 
    'site12': "UMB",
    #'site15': "UPMC", 
    'site17': "UVM",
    'site21': "YALE",
    'site05': "MUSC", 
    'site03': "FIU", 
    'site11': "UFL", 
    'site19': "VCU", 
}

geo_order_siemens = {
    'site02': "CUB", 
    'site16': "UTAH",
    'site09': "UCLA", 
    'site06': "OHSU", 
    'site14': "UMN", 
    'site20': "WUSTL",
    'site07': "ROC", 
    'site12': "UMB",
    #'site15': "UPMC", 
    'site21': "YALE",
    'site05': "MUSC", 
    'site03': "FIU", 
    'site11': "UFL", 
}


## Childhood FC related to PM2.5 exposure
sources = {
    'F1': 'Crustal Materials', # no exposure-related FC
    'F2': 'Ammonium Sulfates', # no exposure-related FC
    'F3': 'Biomass Burning',
    'F4': 'Traffic Emissions',
    'F5': 'Ammonium Nitrates',
    'F6': 'Industrial Fuel',
    'reshist_addr1_pm252016aa': 'PM2.5'
}

NETWORKS = [
    'vs',
    'ad', 
    'smh', 
    'smm',  
    'dla',
    'vta',
    #'n', 
    'rspltp', 
    'ca', 
    'cgc', 
    'sa',
    'fo',
    'dt', 
]

locs = pd.read_csv('site_geography_MRI.csv', index_col=0)
corr_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-corrs_pm25.pkl'))
sens_corr_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-corrs_sensitivity.pkl'))
delta_corr_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-delta-corrs_pm25.pkl'))

corr_dfs = {
    'base': corr_df,
    'sens': sens_corr_df,
    'delta': delta_corr_df
}


delta_df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "delta_rsFC-rci_abs-siemens.pkl")).dropna()
base_df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'data_qcd_base.pkl'))

source_stat = pd.DataFrame(
    dtype=float,
    index=sources.values(),
    columns=pd.MultiIndex.from_product(
        [['cross-sectional', 'longitudinal'], ['mean', 'std']]
    )
)
for source in sources.keys():
    source_name = sources[source]
    source_stat.at[source_name, ('cross-sectional', 'mean')] = base_df[source].mean()
    source_stat.at[source_name, ('cross-sectional', 'std')] = base_df[source].std()
    source_stat.at[source_name, ('longitudinal', 'mean')] = delta_df[source].mean()
    source_stat.at[source_name, ('longitudinal', 'std')] = delta_df[source].std()
source_stat.to_csv(join(PROJ_DIR, OUTP_DIR, 'exposure_source_descriptives.csv'))
### source correlations for cross-sectional and longitudinal samples
# cross-sectional
tril = np.tril(base_df[sources.keys()].corr(), k=-1)
tril = pd.DataFrame(tril, index=sources.values(), columns=sources.keys()).replace({0:np.nan})
fig,ax = plt.subplots(figsize=(7,5))
g = sns.heatmap(tril, cmap='RdBu_r', annot=True, fmt='.2f', linewidths=1, square=True, center=0, ax=ax)
g.set_xticklabels(list(sources.keys())[:-1] + ['PM2.5'])
g.tick_params(axis='y', labelrotation=0)
fig.savefig(
    join(PROJ_DIR, FIGS_DIR, 'supplemental_source_corrs.png'),
    dpi=400,
    bbox_inches='tight'
)
# longitudinal
tril = np.tril(delta_df[sources.keys()].corr(), k=-1)
tril = pd.DataFrame(tril, index=sources.values(), columns=sources.keys()).replace({0:np.nan})
fig,ax = plt.subplots(figsize=(7,5))
g = sns.heatmap(tril, cmap='RdBu_r', annot=True, fmt='.2f', linewidths=1, square=True, center=0, ax=ax)
g.set_xticklabels(list(sources.keys())[:-1] + ['PM2.5'])
g.tick_params(axis='y', labelrotation=0)
fig.savefig(
    join(PROJ_DIR, FIGS_DIR, 'supplemental_source_corrs-delta.png'),
    dpi=400,
    bbox_inches='tight'
)

###################################################
 
#ntwk_sum = pd.DataFrame(
#    dtype=float,
#    index=NETWORKS,
#    columns=sources.keys()
#)

#delta_ntwk_sum = pd.DataFrame(
#    dtype=float,
#    index=NETWORKS,
#    columns=sources.keys()
#)

#both_ntwk_sum = pd.DataFrame(
#    dtype=float,
#    index=NETWORKS,
#    columns=sources.keys()
#)

base_model = pd.read_pickle(
    join(PROJ_DIR, OUTP_DIR, 'model_stats_base.pkl')
)
delta_model = pd.read_pickle(
    join(PROJ_DIR, OUTP_DIR, 'model_stats_delta.pkl')
)


models_x_contributions = pd.DataFrame(
    dtype=float,
    index=pd.MultiIndex.from_product([site_names.keys(), sources.keys()]),
)

for source in sources.keys():
    mse_temp = mse_df.loc[source]
    delta_mse = delta_mse_df.loc[source]

    #for network in NETWORKS:
    #    ntwk_sum.at[network, source] = np.sum(np.abs(mse_temp.filter(like=network).mean()) > 0.01)
    #    delta_ntwk_sum.at[network, source] = np.sum(delta_temp.filter(like=network).mean() > 0.01)
    #    both_ntwk_sum.at[network, source] = np.sum(mse_temp.filter(like=network).mean() + delta_temp.filter(like=network).mean() > 0.01)
    mse_temp = mse_temp.mean(axis=1)
    #base_corrs = mse_df.loc[source].mean().dropna()
    #delta_corrs = delta_mse_df.loc[source].mean().dropna()
    #comparison[source]['both'] = list(set(base_corrs.index) & set(delta_corrs.index))
    #comparison[source]['base_only'] = list(set(base_corrs.index) - set(delta_corrs.index))
    #comparison[source]['delta_only'] = list(set(delta_corrs.index) - set(base_corrs.index))
    for site in site_names.keys():
        temp = pd.DataFrame(dtype=float)
        site_df = base_df[base_df['site_id_l'] == site]
        source_mean = site_df[source].mean()
        models_x_contributions.at[(site, source), 'female_percent'] = len(site_df[site_df['demo_sex_v2_bl'] == 'Female'].index) / len(site_df.index)
        models_x_contributions.at[(site, source), 'high_income_pct'] = len(site_df[site_df['household_income_4bins_bl'] == '[≥100K]'].index) / len(site_df.index)
        models_x_contributions.at[(site, source), 'low_income_pct'] = len(site_df[site_df['household_income_4bins_bl'] == '[<50K]'].index) / len(site_df.index)
        models_x_contributions.at[(site, source), 'scanner_manufacturer'] = site_df['mri_info_manufacturer'].unique()[0]
        models_x_contributions.at[(site, source), 'nonhispwhite_pct'] = len(site_df[site_df['race_ethnicity_c_bl'] == 'White'].index) / len(site_df.index)

        models_x_contributions.at[(site, source), f'contribution_mean'] = source_mean
        models_x_contributions.at[(site, source), f'contribution_std'] = site_df[source].std()
        models_x_contributions.at[(site, source), f'rmse_in'] = base_model.loc[site][(source, 'rmse_train')]
        models_x_contributions.at[(site, source), f'rmse_out'] = base_model.loc[site][(source, 'rmse_test')]
        models_x_contributions.at[(site, source), f'rsq_in'] = base_model.loc[site][(source, 'rsq_train')]
        models_x_contributions.at[(site, source), f'rsq_out'] = base_model.loc[site][(source, 'rsq_test')]
        models_x_contributions.at[(site, source), 'source'] = source
        models_x_contributions.at[(site, source), 'pm25_mean'] = site_df['reshist_addr1_pm252016aa'].mean()
        #models_x_contributions.at[site, (source, f'{source}-proportion')] = mse_temp.loc[site] / source_mean
        #models_x_contributions.at[site, (source, 'table_col')] = f'{np.round(source_mean, 2)} ± {np.round(site_df[source].std(), 2)}'
#with open(join(PROJ_DIR, OUTP_DIR, 'base_v_delta-comparisons.json'), 'w') as f:
#    json.dump(comparison, f)
models_x_contributions["source"] = models_x_contributions.index.get_level_values(1)

models_x_contributions.to_csv(
    join(PROJ_DIR, OUTP_DIR, 'mse x contributions.csv')
)

factors = {
    "contribution_mean": "Exposure Mean",
    "contribution_std": "Exposure Standard Deviation",
    "female_percent": "Percent Female",
    "high_income_pct": "Percent High Income (≥$100k)",
    "low_income_pct": "Percent Low Income (≥$100k)",
    "nonhispwhite_pct": "Percent Non-Hispanic White",
    "pm25_mean": "Mean PM2.5"
}

###################
sns.set(context='paper', style='white', palette='husl', font_scale=2)
mse_corrs = pd.DataFrame(
    dtype=float,
    columns=pd.MultiIndex.from_product([sources.values(), ['r', 'p']]),
    index=factors.values()
)

for factor in factors.keys():
    row = factors[factor]
    for source in sources.keys():
        col = sources[source]
        temp3 = models_x_contributions.xs(source, level=1)
        r, p = spearmanr(
                temp3[factor], 
                temp3['rmse_out'],
                nan_policy='omit'
            )
        mse_corrs.at[row, (col, 'r')] = np.round(r,3)
        mse_corrs.at[row, (col, 'p')] = np.round(p,4)
    g = sns.lmplot(
        x=factor, 
        y="rmse_out", 
        data=models_x_contributions, 
        col='source',
        aspect=1,
        sharex=False,
        sharey=False
    )
    def annotate(data, **kws):
        r, p = spearmanr(
            data[factor], 
            data['rmse_out'],
            nan_policy='omit'
        )
        ax = plt.gca()
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                transform=ax.transAxes)
        
    g.map_dataframe(annotate)
    g.savefig(
        join(PROJ_DIR, FIGS_DIR, f"{factor}-mse_out-plots.png"),
        dpi=600, bbox_inches='tight'
    )
mse_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'mse_corrs_base.csv'))

rsq_corrs = pd.DataFrame(
    dtype=float,
    columns=pd.MultiIndex.from_product([sources.values(), ['r', 'p']]),
    index=factors.values()
)

for factor in factors.keys():
    row = factors[factor]
    for source in sources.keys():
        col = sources[source]
        temp3 = models_x_contributions.xs(source, level=1)
        r, p = spearmanr(
                temp3[factor], 
                temp3['rsq_out'],
                nan_policy='omit'
            )
        rsq_corrs.at[row, (col, 'r')] = np.round(r,3)
        rsq_corrs.at[row, (col, 'p')] = np.round(p,4)
    g = sns.lmplot(
        x=factor, 
        y="rsq_out", 
        data=models_x_contributions, 
        col='source',
        aspect=1,
        sharex=False,
        sharey=False
    )
    def annotate(data, **kws):
        r, p = spearmanr(
            data[factor], 
            data['rsq_out'],
            nan_policy='omit'
        )
        ax = plt.gca()
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                transform=ax.transAxes)
        
    g.map_dataframe(annotate)
    g.savefig(
        join(PROJ_DIR, FIGS_DIR, f"{factor}-rsq_out-plots.png"),
        dpi=600, bbox_inches='tight'
    )
rsq_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'rsq_corrs_base.csv'))


g = sns.swarmplot(models_x_contributions, x='source', y='model_performance',hue='scanner_manufacturer')
g.set_xticklabels(list(sources.keys())[:-1] + ['PM2.5'])
g.savefig(
    join(PROJ_DIR, FIGS_DIR, f"scanner_performance_plots.png"),
    dpi=600, bbox_inches='tight'
)

###############################
# now for delta models
models_x_contributions = pd.DataFrame(
    dtype=float,
    index=pd.MultiIndex.from_product([site_names.keys(), sources.keys()]),
)

for source in sources.keys():
    mse_temp = mse_df.loc[source]
    delta_mse = delta_mse_df.loc[source]

    #for network in NETWORKS:
    #    ntwk_sum.at[network, source] = np.sum(np.abs(mse_temp.filter(like=network).mean()) > 0.01)
    #    delta_ntwk_sum.at[network, source] = np.sum(delta_temp.filter(like=network).mean() > 0.01)
    #    both_ntwk_sum.at[network, source] = np.sum(mse_temp.filter(like=network).mean() + delta_temp.filter(like=network).mean() > 0.01)
    mse_temp = mse_temp.mean(axis=1)
    delta_mse = delta_mse.mean(axis=1)
    #base_corrs = mse_df.loc[source].mean().dropna()
    #delta_corrs = delta_mse_df.loc[source].mean().dropna()
    #comparison[source]['both'] = list(set(base_corrs.index) & set(delta_corrs.index))
    #comparison[source]['base_only'] = list(set(base_corrs.index) - set(delta_corrs.index))
    #comparison[source]['delta_only'] = list(set(delta_corrs.index) - set(base_corrs.index))
    for site in geo_order_siemens.keys():
        temp = pd.DataFrame(dtype=float)
        site_df = delta_df[delta_df['site_id_l'] == site]
        source_mean = site_df[source].mean()
        models_x_contributions.at[(site, source), 'female_percent'] = len(site_df[site_df['demo_sex_v2_bl'] == 'Female'].index) / len(site_df.index)
        models_x_contributions.at[(site, source), 'high_income_pct'] = len(site_df[site_df['household_income_4bins_bl'] == '[≥100K]'].index) / len(site_df.index)
        models_x_contributions.at[(site, source), 'low_income_pct'] = len(site_df[site_df['household_income_4bins_bl'] == '[<50K]'].index) / len(site_df.index)
        models_x_contributions.at[(site, source), 'scanner_manufacturer'] = site_df['mri_info_manufacturer'].unique()[0]
        models_x_contributions.at[(site, source), 'nonhispwhite_pct'] = len(site_df[site_df['race_ethnicity_c_bl'] == 'White'].index) / len(site_df.index)

        models_x_contributions.at[(site, source), f'contribution_mean'] = source_mean
        models_x_contributions.at[(site, source), f'contribution_std'] = site_df[source].std()
        models_x_contributions.at[(site, source), f'rmse_in'] = delta_model.loc[site][(source, 'rmse_train')]
        models_x_contributions.at[(site, source), f'rmse_out'] = delta_model.loc[site][(source, 'rmse_test')]
        models_x_contributions.at[(site, source), f'rsq_in'] = delta_model.loc[site][(source, 'rsq_train')]
        models_x_contributions.at[(site, source), f'rsq_out'] = delta_model.loc[site][(source, 'rsq_test')]
        models_x_contributions.at[(site, source), 'source'] = source
        models_x_contributions.at[(site, source), 'pm25_mean'] = site_df['reshist_addr1_pm252016aa'].mean()
        #models_x_contributions.at[site, (source, f'{source}-proportion')] = mse_temp.loc[site] / source_mean
        #models_x_contributions.at[site, (source, 'table_col')] = f'{np.round(source_mean, 2)} ± {np.round(site_df[source].std(), 2)}'
#with open(join(PROJ_DIR, OUTP_DIR, 'base_v_delta-comparisons.json'), 'w') as f:
#    json.dump(comparison, f)
models_x_contributions["source"] = models_x_contributions.index.get_level_values(1)

models_x_contributions.to_csv(
    join(PROJ_DIR, OUTP_DIR, 'mse x contributions delta.csv')
)

###################
sns.set(context='paper', style='white', palette='husl', font_scale=2)
mse_corrs = pd.DataFrame(
    dtype=float,
    columns=pd.MultiIndex.from_product([sources.values(), ['r', 'p']]),
    index=factors.values()
)

for factor in factors.keys():
    row = factors[factor]
    for source in sources.keys():
        col = sources[source]
        temp3 = models_x_contributions.xs(source, level=1)
        r, p = spearmanr(
                temp3[factor], 
                temp3['rmse_out'],
                nan_policy='omit'
            )
        mse_corrs.at[row, (col, 'r')] = np.round(r,3)
        mse_corrs.at[row, (col, 'p')] = np.round(p,4)
    g = sns.lmplot(
        x=factor, 
        y="rmse_out", 
        data=models_x_contributions, 
        col='source',
        aspect=1,
        sharex=False,
        sharey=False
    )
    def annotate(data, **kws):
        r, p = spearmanr(
            data[factor], 
            data['rmse_out'],
            nan_policy='omit'
        )
        ax = plt.gca()
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                transform=ax.transAxes)
        
    g.map_dataframe(annotate)
    g.savefig(
        join(PROJ_DIR, FIGS_DIR, f"{factor}-mse_out_delta-plots.png"),
        dpi=600, bbox_inches='tight'
    )
mse_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'mse_corrs_delta.csv'))

rsq_corrs = pd.DataFrame(
    dtype=float,
    columns=pd.MultiIndex.from_product([sources.values(), ['r', 'p']]),
    index=factors.values()
)

for factor in factors.keys():
    row = factors[factor]
    for source in sources.keys():
        col = sources[source]
        temp3 = models_x_contributions.xs(source, level=1)
        r, p = spearmanr(
                temp3[factor], 
                temp3['rsq_out'],
                nan_policy='omit'
            )
        rsq_corrs.at[row, (col, 'r')] = np.round(r,3)
        rsq_corrs.at[row, (col, 'p')] = np.round(p,4)
    g = sns.lmplot(
        x=factor, 
        y="rsq_out", 
        data=models_x_contributions, 
        col='source',
        aspect=1,
        sharex=False,
        sharey=False
    )
    def annotate(data, **kws):
        r, p = spearmanr(
            data[factor], 
            data['rsq_out'],
            nan_policy='omit'
        )
        ax = plt.gca()
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                transform=ax.transAxes)
        
    g.map_dataframe(annotate)
    g.savefig(
        join(PROJ_DIR, FIGS_DIR, f"{factor}-rsq_out_delta-plots.png"),
        dpi=600, bbox_inches='tight'
    )
rsq_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'rsq_corrs_delta.csv'))


g = sns.swarmplot(models_x_contributions, x='source', y='model_performance',hue='scanner_manufacturer')
g.set_xticklabels(list(sources.keys())[:-1] + ['PM2.5'])
g.savefig(
    join(PROJ_DIR, FIGS_DIR, f"scanner_performance_plots.png"),
    dpi=600, bbox_inches='tight'
)

####################
########################
sns.set(context='paper', style='white', palette='husl', font_scale=1.5)

describe = pd.DataFrame(
    index=sources.keys(),
    columns=['mean', 'sdev']
)
for source in sources.keys():
    describe.at[source, 'mean'] = mse_df.mean(axis=1).loc[source].mean()
    describe.at[source, 'sdev'] = mse_df.mean(axis=1).loc[source].std()
describe.to_csv(join(PROJ_DIR, OUTP_DIR, 'CPM-mse_by_source.csv'))

delta_describe = pd.DataFrame(
    index=sources.keys(),
    columns=['mean', 'sdev']
)
for source in sources.keys():
    delta_describe.at[source, 'mean'] = delta_mse_df.mean(axis=1).loc[source].mean()
    delta_describe.at[source, 'sdev'] = delta_mse_df.mean(axis=1).loc[source].std()
delta_describe.to_csv(join(PROJ_DIR, OUTP_DIR, 'CPM-delta-mse_by_source.csv'))

rsq_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-rsq_pm25.pkl'))
delta_rsq_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-delta-rsq_pm25.pkl'))

describe = pd.DataFrame(
    index=sources.keys(),
    columns=['mean', 'sdev']
)
for source in sources.keys():
    describe.at[source, 'mean'] = rsq_df.mean(axis=1).loc[source].mean()
    describe.at[source, 'sdev'] = rsq_df.mean(axis=1).loc[source].std()
describe.to_csv(join(PROJ_DIR, OUTP_DIR, 'CPM-rsq_by_source.csv'))

delta_describe = pd.DataFrame(
    index=sources.keys(),
    columns=['mean', 'sdev']
)
for source in sources.keys():
    delta_describe.at[source, 'mean'] = delta_rsq_df.mean(axis=1).loc[source].mean()
    delta_describe.at[source, 'sdev'] = delta_rsq_df.mean(axis=1).loc[source].std()
delta_describe.to_csv(join(PROJ_DIR, OUTP_DIR, 'CPM-delta-rsq_by_source.csv'))


source_pal = sns.color_palette(
    [
        '#2B9ADE', 
        '#785EF0', 
        '#DC267F', 
        '#FE6100', 
        '#FFB000', 
        '#858686',
        "#111111"
    ]
)

sites = pd.DataFrame(dtype=str, index=range(21)).drop(18)
sites = pd.concat([sites, pd.Series(index_to_site, name='site_nums')], axis=1).drop(18)
for site in sites.index:
    site_name = site_names[sites.loc[site]['site_nums']]
    sites.at[site, 'site_names'] = site_name
    sites.at[site, 'site_cities'] = site_cities[sites.loc[site]['site_nums']]
    for region in regions.keys():
        if site_name in regions[region]:
            sites.at[site, 'region'] = region#\

#locs = gpd.tools.geocode(sites.site_cities, provider='nominatim', user_agent="klb0036")
locs = pd.DataFrame()
for city in sites.site_cities:
    temp = gpd.tools.geocode(city, provider='nominatim', user_agent='bottfam')
    locs = pd.concat(
        [
            locs,
            temp
        ],
        axis=0
    )
locs.index = sites.index
locs = pd.concat([locs, sites], axis=1)
locs.to_pickle(
    join(PROJ_DIR, DATA_DIR, 'geocoded_sites.pkl')
)
locs.to_file("geocoded_sites.geojson", driver='GeoJSON')


for i in locs.index:
    for source in sources:
        site = locs.loc[i]['site_nums']
        avg = base_df[base_df['site_id_l'] == site][source].mean()
        std = base_df[base_df['site_id_l'] == site][source].std()
        locs.at[i, f'{source}_mean'] = avg
        locs.at[i, f'{source}_sdev'] = std
        if site == 'site15':
            mri = 'SIEMENS'
        else:
            mri = base_df[base_df['site_id_l'] == site]['mri_info_manufacturer'].unique()[0]
        locs.at[i, 'MRI'] = mri


#for i in locs.index:
#    for source in sources:
#        site = locs.loc[i]['site_nums']
#        avg = df[df['site_id_l'] == site][source].mean()
#        std = df[df['site_id_l'] == site][source].std()
#        locs.at[i, f'{source}_mean'] = avg
#        locs.at[i, f'{source}_sdev'] = std

performance_corrs = pd.DataFrame(
  dtype=float,
  columns=pd.MultiIndex.from_product([['mean', 'dev'],['r', 'p']]),
  index=pd.MultiIndex.from_product([sources.keys(), ['base', 'delta']])
)

locs.index = locs['site_nums']
cmap2 = sns.blend_palette(['#009E73','#F0E442', '#D55E00'], as_cmap=True)
for source in sources.keys():
    source_name = sources[source]
    temp = mse_df.loc[source].mean(axis=1)
    temp.name = source_name
    
    #temp.index = temp['site_nums']
    temp2 = pd.concat([locs, temp], axis=1)
    #temp2 = gpd.GeoDataFrame(temp2, crs='ESRI:102003')
    # create map of all states except AK and HI in the main map axis
    #fig,ax = plt.subplots(ncols=3, figsize=(11,4))
    fig,ax = plt.subplots(nrows=2,figsize=(4,3), height_ratios=[4,.5])
    usa = usa.to_crs("ESRI:102003")
    ax0 = usa.plot(color='.9', edgecolor='0.9', ax=ax[0]
                    )
    ax0.axis('off')
    ax0.set_title(source_name)
    temp2.to_crs("ESRI:102003").plot(
        ax=ax0, 
        column=source_name, 
        legend=True,
        cmap=cmap2,
        
        markersize=30,
        #norm=colors.CenteredNorm(),
        #markersize=temp2[f'{source}_mean'] ** 2 * 25,
        
    )
    legend = ax0.get_legend().remove()
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(
        vmin=np.round(temp2[source_name].min(),2), 
        vmax=np.round(temp2[source_name].max(),2)
    )

    cb1 = mpl.colorbar.ColorbarBase(ax[1], cmap=cmap2,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('RMSE')
    #ax1 = usa.plot(color='0.8', edgecolor='0.8', ax=ax[1])
    #ax1.axis('off')
    r,p = spearmanr(temp2[[source_name, f'{source}_mean']].dropna())
    performance_corrs.at[(source, 'base'), ('mean','r')] = r
    performance_corrs.at[(source, 'base'), ('mean','p')] = p
    #ax[1].set_title(f'r = {np.round(r, 3)}, p = {np.round(p,4)}')
    #temp2.to_crs("ESRI:102003").plot(
    #    ax=ax1, 
    #    column=f'{source}_mean', 
    #    legend=True,
    #    cmap='viridis',
    #    legend_kwds={
    #        #"label": "$r_S$: actual vs. predicted", 
    #        "label": "Exposure Mean",
    #        "orientation": "horizontal", 
    #        "shrink": 0.75
    #        },
    #)

    #ax2 = usa.plot(color='0.8', edgecolor='0.8', ax=ax[2])
    #ax2.axis('off')
    r,p = spearmanr(temp2[[source_name, f'{source}_sdev']].dropna())
    performance_corrs.at[(source, 'base'), ('dev','r')] = r
    performance_corrs.at[(source, 'base'), ('dev','p')] = p
    #ax[2].set_title(f'r = {np.round(r, 3)}, p = {np.round(p,4)}')
    #temp2.to_crs("ESRI:102003").plot(
    #    ax=ax2, 
    #    column=f'{source}_sdev', 
    #    legend=True,
    #    cmap='viridis',
    #    legend_kwds={
    #        #"label": "$r_S$: actual vs. predicted", 
    #        "label": "Exposure Variance",
    #        "orientation": "horizontal", 
    #        "shrink": 0.75
    #        },
    #)
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f"{source}_base_maps.png"), dpi=600, bbox_inches='tight')


long_mse = pd.DataFrame(mse_df.mean(axis=1), index=mse_df.index, columns=['RMSE'])
long_mse['source'] = long_mse.index.get_level_values(0)

fig,ax = plt.subplots(figsize=(5,3))
g = sns.kdeplot(
    long_mse,
    hue='source',
    x='RMSE', 
    fill=True, 
    palette=source_pal, 
    ax=ax,
    legend=True
)
g.get_legend().remove()

ax.set_xlabel('Root Mean Squared Error', fontsize=14)
ax.set_xlim(-0.01,1.25)
sns.despine()
#ax.axvline(0, lw=2, ls='--', color='#333333', alpha=0.4)

fig.savefig(
    join(PROJ_DIR, FIGS_DIR, f'CPM-base-performance_mse.png'), 
    dpi=600, 
    bbox_inches='tight'
)

for source in sources.keys():
    source_name = sources[source]
    temp = delta_mse_df.loc[source].mean(axis=1)
    temp.name = source_name
    
    #temp.index = temp['site_nums']
    temp2 = pd.concat([locs, temp], axis=1)
    # create map of all states except AK and HI in the main map axis
    #fig,ax = plt.subplots(ncols=3, figsize=(11,4))
    fig,ax = plt.subplots(nrows=2,figsize=(4,3), height_ratios=[4,.5])
    usa = usa.to_crs("ESRI:102003")
    ax0 = usa.plot(color='0.9', edgecolor='0.9', ax=ax[0]
                    )
    ax0.axis('off')
    ax0.set_title(source_name)
    temp2.to_crs("ESRI:102003").plot(
        ax=ax0, 
        column=source_name, 
        legend=True,
        cmap=cmap2,
        markersize=30,
        #norm=colors.CenteredNorm(),
        #markersize=temp2[f'{source}_mean'] ** 2 * 25,
        
    )
    legend = ax0.get_legend().remove()
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(
        vmin=np.round(temp2[source_name].min(),2), 
        vmax=np.round(temp2[source_name].max(),2)
    )
    
    fmt = lambda x, pos: np.round(x,2)
    cb1 = mpl.colorbar.ColorbarBase(ax[1], cmap=cmap2,
                                    norm=norm,
                                    orientation='horizontal',
                                    format=FuncFormatter(fmt)
                                    )
    cb1.set_label('RMSE')
    #ax1 = usa.plot(color='0.8', edgecolor='0.8', ax=ax[1])
    #ax1.axis('off')
    r,p = spearmanr(temp2[[source_name, f'{source}_mean']].dropna())
    performance_corrs.at[(source, 'delta'), ('mean','r')] = r
    performance_corrs.at[(source, 'delta'), ('mean','p')] = p
    #ax[1].set_title(f'r = {np.round(delta, 3)}, p = {np.round(p,4)}')
    #temp2.to_crs("ESRI:102003").plot(
    #    ax=ax1, 
    #    column=f'{source}_mean', 
    #    legend=True,
    #    cmap='viridis',
    #    legend_kwds={
    #        #"label": "$r_S$: actual vs. predicted", 
    #        "label": "Exposure Mean",
    #        "orientation": "horizontal", 
    #        "shrink": 0.75
    #        },
    #)

    #ax2 = usa.plot(color='0.8', edgecolor='0.8', ax=ax[2])
    #ax2.axis('off')
    r,p = spearmanr(temp2[[source_name, f'{source}_sdev']].dropna())
    performance_corrs.at[(source, 'delta'), ('dev','r')] = r
    performance_corrs.at[(source, 'delta'), ('dev','p')] = p
    #ax[2].set_title(f'r = {np.round(r, 3)}, p = {np.round(p,4)}')
    #temp2.to_crs("ESRI:102003").plot(
    #    ax=ax2, 
    #    column=f'{source}_sdev', 
    #    legend=True,
    #    cmap='viridis',
    #    legend_kwds={
    #        #"label": "$r_S$: actual vs. predicted", 
    #        "label": "Exposure Variance",
    #        "orientation": "horizontal", 
    #        "shrink": 0.75
    #        },
    #)
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f"{source}_delta_maps.png"), dpi=600, bbox_inches='tight')


long_delta_mse = pd.DataFrame(delta_mse_df.mean(axis=1), index=delta_mse_df.index, columns=['RMSE'])
long_delta_mse['Source'] = long_delta_mse.index.get_level_values(0)
long_delta_mse = long_delta_mse.replace(sources)

fig,ax = plt.subplots(figsize=(5,3))
g = sns.kdeplot(
    long_delta_mse,
    hue='Source',
    x='RMSE', 
    fill=True, 
    palette=source_pal, 
    ax=ax,
    legend=True
)
legend = g.get_legend()
legend.remove()
ax.set_xlabel('Root Mean Squared Error', fontsize=14)
ax.set_xlim(-0.01,1.25)
sns.despine()
#ax.axvline(0, lw=2, ls='--', color='#333333', alpha=0.4)

fig.savefig(
    join(PROJ_DIR, FIGS_DIR, f'CPM-delta-performance_mse.png'), 
    dpi=600, 
    bbox_inches='tight'
)

#performance_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'mse_x_exposure-correlations_by_site.csv'))
