import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter

from os.path import join
from scipy.stats import spearmanr
from scipy.spatial.distance import dice, jaccard
from nilearn import plotting, surface, datasets
from geopy.geocoders import Nominatim
import geopandas as gpd
import geodatasets

import math
import json
import matplotlib as mpl

gdf = gpd.read_file('/Users/katherine.b/Dropbox/Mac/Downloads/cb_2018_us_state_500k(1)')
gdf.head()

usa = gpd.read_file(geodatasets.get_path('geoda.natregimes'))

sns.set(context='paper', style='white', palette='husl', font_scale=1.5)


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
    'F6': 'Industrial Fuel'
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

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'data_qcd_delta.pkl'))
mse_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-mse.pkl'))
delta_mse_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-delta-mse.pkl'))

mse_df.T.mean().unstack(level=1).T.rename(geo_order, axis=0).loc[geo_order.values()].to_csv(
    join(PROJ_DIR, OUTP_DIR, 'CPM-mse_by_site.csv')
)

delta_mse_df.T.mean().unstack(level=1).T.rename(geo_order_siemens, axis=0).loc[geo_order_siemens.values()].to_csv(
    join(PROJ_DIR, OUTP_DIR, 'CPM-delta-mse_by_site.csv')
)
#locs = pd.read_csv('site_geography_MRI.csv', index_col=0)
corr_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-corrs.pkl'))
delta_corr_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-delta-corrs.pkl'))


models_x_contributions = pd.DataFrame(
    dtype=float,
    index=site_names.keys(),
    columns=pd.MultiIndex.from_product(
        [
            list(sources.keys()),
            ['table_col','contribution_mean', 'contribution_std', 'model_performance', 'proportion']
        ]
    )
)
dat = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'data_qcd.pkl'))

tril = np.tril(dat.filter(like='F').corr(), k=-1)
tril = pd.DataFrame(tril, index=sources.values(), columns=sources.keys()).replace({0:np.nan})
fig,ax = plt.subplots()
g = sns.heatmap(tril, cmap='seismic', annot=True, square=True, center=0, ax=ax)
#g.set_yticklabels(sources.values())
g.tick_params(axis='y', labelrotation=0)
fig.savefig(
    join(PROJ_DIR, FIGS_DIR, 'supplemental_source_corrs.png'),
    dpi=400,
    bbox_inches='tight'
)
ntwk_sum = pd.DataFrame(
    dtype=float,
    index=NETWORKS,
    columns=sources.keys()
)

delta_ntwk_sum = pd.DataFrame(
    dtype=float,
    index=NETWORKS,
    columns=sources.keys()
)


comparison = {}
for source in sources.keys():
    comparison[source] = {}
    mse_temp = mse_df.loc[source]
    delta_temp = delta_mse_df.loc[source]

    corr_temp = corr_df.loc[source]
    delta_corr_temp = delta_corr_df.loc[source]

    ########################
    delta_bin = delta_corr_temp.isna().replace(
        {
            True: '',
            False: 'Δ'
        }
    )
    corr_bin = corr_temp.isna().replace(
        {
            True: '',
            False: '$Τ_1$'
        }
    )
    both = corr_bin + delta_bin
    both = both.replace(
        {
            '$Τ_1$Δ': '$Τ_1,\Delta$'
        }
    )
    num_temp = (corr_temp.fillna(0) + delta_corr_temp.fillna(0)) / 2
    avg_df = pd.DataFrame(
        dtype=float,
        index=NETWORKS,
        columns=NETWORKS
    )
    mode_df = pd.DataFrame(
        dtype=str,
        index=NETWORKS,
        columns=NETWORKS
    )
    for var in num_temp.columns:
        ntwk1 = var.split('_')[3]
        ntwk2 = var.split('_')[5]
        avg_df.at[ntwk1, ntwk2] = num_temp[var].mean()
        avg_df.at[ntwk2, ntwk1] = num_temp[var].mean()
        mode_df.at[ntwk1, ntwk2] = both[var].mode()[0]
        mode_df.at[ntwk2, ntwk1] = both[var].mode()[0]
    fig,ax = plt.subplots(figsize=(7,7))
    sns.heatmap(avg_df, annot=mode_df, fmt='', cmap='seismic', center=0, square=True,ax=ax),
    ax.set_title(sources[source])
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'{source}-timepoints_avg.png'),
        dpi=400,
        bbox_inches='tight'
    )
    for network in NETWORKS:
        ntwk_sum.at[network, source] = np.sum(np.abs(mse_temp.filter(like=network).mean()) > 0.01)
        delta_ntwk_sum.at[network, source] = np.sum(delta_temp.filter(like=network).mean() > 0.01)
    mse_temp = mse_temp.mean(axis=1)
    base_corrs = mse_df.loc[source].mean().dropna()
    delta_corrs = delta_mse_df.loc[source].mean().dropna()
    comparison[source]['both'] = list(set(base_corrs.index) & set(delta_corrs.index))
    comparison[source]['base_only'] = list(set(base_corrs.index) - set(delta_corrs.index))
    comparison[source]['delta_only'] = list(set(delta_corrs.index) - set(base_corrs.index))
    for site in site_names.keys():
        site_df = dat[dat['site_id_l'] == site]
        source_mean = site_df[source].mean()
        models_x_contributions.at[site, (source, 'contribution_mean')] = source_mean
        models_x_contributions.at[site, (source, 'contribution_std')] = site_df[source].std()
        models_x_contributions.at[site, (source, 'model_performance')] = mse_temp.loc[site]
        models_x_contributions.at[site, (source, 'proportion')] = mse_temp.loc[site] / source_mean
        models_x_contributions.at[site, (source, 'table_col')] = f'{np.round(source_mean, 2)} ± {np.round(site_df[source].std(), 2)}'

with open(join(PROJ_DIR, OUTP_DIR, 'base_v_delta-comparisons.json'), 'w') as f:
    json.dump(comparison, f)

models_x_contributions.to_csv(
    join(PROJ_DIR, OUTP_DIR, 'mse x contributions.csv')
)


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

rsq_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-rsq.pkl'))
delta_rsq_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-delta-rsq.pkl'))

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
        '#858686'
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
    temp = gpd.tools.geocode(city, provider='nominatim', user_agent='bottfam2')
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
        avg = df[df['site_id_l'] == site][source].mean()
        std = df[df['site_id_l'] == site][source].std()
        locs.at[i, f'{source}_mean'] = avg
        locs.at[i, f'{source}_sdev'] = std

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

ax.set_xlabel('Root Mean Squared Error', fontsize=24)
ax.set_xlim(-0.01,0.61)
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
ax.set_xlabel('Root Mean Squared Error', fontsize=24)
ax.set_xlim(-0.01,0.61)
sns.despine()
#ax.axvline(0, lw=2, ls='--', color='#333333', alpha=0.4)

fig.savefig(
    join(PROJ_DIR, FIGS_DIR, f'CPM-delta-performance_mse.png'), 
    dpi=600, 
    bbox_inches='tight'
)

performance_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'mse_x_exposure-correlations_by_site.csv'))

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

corr_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-corrs.pkl'))
delta_corr_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'CPM-delta-corrs.pkl'))

for source in sources.keys():
    avg_corrs = corr_df.fillna(0).loc[source].mean()
    corrmat = pd.DataFrame(
        dtype=float,
        index=NETWORKS.values(),
        columns=NETWORKS.keys()
    )
    avg_corrs = avg_corrs.replace(
        {
            0: 999
        }
    )
    for varname in avg_corrs.index:
        network1 = varname.split('_')[3]
        network2 = varname.split('_')[5]
        if np.abs(avg_corrs[varname]) > 0.01:
            #print(avg_corrs[varname])
            corrmat.at[NETWORKS[network1], network2] = avg_corrs[varname]
            corrmat.at[NETWORKS[network2], network1] = avg_corrs[varname]
        else:
            corrmat.at[NETWORKS[network1], network2] = 999
            corrmat.at[NETWORKS[network2], network1] = 999
    triangle = pd.DataFrame(
        np.tril(corrmat),
        index=corrmat.index,
        columns=corrmat.columns
    )
    triangle = triangle.replace(
        {
            0: np.nan,
            999: 0
        }
    )
    fig,ax = plt.subplots(figsize=(4,4))
    g = sns.heatmap(
        triangle, 
        cmap='RdBu_r', 
        center=0, 
        square=True, 
        ax=ax
    )
    ax.set_xticklabels([NTWKS[i] for i in NETWORKS.keys()])
    ax.set_title(sources[source])
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'CPM-{source}-baseline_corrs.png'),
        dpi=400,
        facecolor='#FFFFFF',
        bbox_inches='tight'
    )

    avg_corrs = delta_corr_df.fillna(0).loc[source].mean()
    corrmat = pd.DataFrame(
        dtype=float,
        index=NETWORKS.values(),
        columns=NETWORKS.keys()
    )
    avg_corrs = avg_corrs.replace(
        {
            0: 999
        }
    )
    for varname in avg_corrs.index:
        network1 = varname.split('_')[3]
        network2 = varname.split('_')[5]
        if np.abs(avg_corrs[varname]) > 0.01:
            #print(avg_corrs[varname])
            corrmat.at[NETWORKS[network1], network2] = avg_corrs[varname]
            corrmat.at[NETWORKS[network2], network1] = avg_corrs[varname]
        else:
            corrmat.at[NETWORKS[network1], network2] = 999
            corrmat.at[NETWORKS[network2], network1] = 999
    triangle = pd.DataFrame(
        np.tril(corrmat),
        index=corrmat.index,
        columns=corrmat.columns
    )
    triangle = triangle.replace(
        {
            0: np.nan,
            999: 0
        }
    )
    fig,ax = plt.subplots(figsize=(4,4))
    g = sns.heatmap(triangle, cmap='RdBu_r', center=0, square=True, ax=ax)
    ax.set_xticklabels([NTWKS[i] for i in NETWORKS.keys()])
    ax.set_title(sources[source])
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'CPM-{source}-delta_corrs.png'),
        dpi=400,
        facecolor='#FFFFFF',
        bbox_inches='tight'
    )
    
