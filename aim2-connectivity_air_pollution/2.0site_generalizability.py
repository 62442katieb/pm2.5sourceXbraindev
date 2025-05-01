import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as colors


from os.path import join
from scipy.stats import spearmanr
from scipy.spatial.distance import dice, jaccard
from nilearn import plotting, surface, datasets
from geopy.geocoders import Nominatim
import geopandas as gpd
import geodatasets

import math

gdf = gpd.read_file('/Users/katherine.b/Dropbox/Mac/Downloads/cb_2018_us_state_500k(1)')
gdf.head()

usa = gpd.read_file(geodatasets.get_path('geoda.natregimes'))

sns.set(context='paper', style='white', palette='husl', font_scale=2)


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
              'site13': "UMICH", 'site14': "UMN", 'site15': "UPMC", 'site16': "UTAH",
              'site17': "UVM", 'site18': "UWM",  'site19': "VCU", 'site20': "WUSTL",
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
    'site15': "Pittsburgh, PA", 
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

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'data_qcd_delta.pkl'))

## Childhood FC related to PM2.5 exposure
sources = {
    'F1': 'Crustal', # no exposure-related FC
    'F2': 'Ammonium Sulfates', # no exposure-related FC
    'F3': 'Biomass Burning',
    'F4': 'Traffic',
    'F5': 'Ammonium Nitrates',
    'F6': 'Industrial Fuel'
}

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

sites = pd.DataFrame(dtype=str, index=range(21))
sites = pd.concat([sites, pd.Series(index_to_site, name='site_nums')], axis=1)
for site in sites.index:
    site_name = site_names[sites.loc[site]['site_nums']]
    sites.at[site, 'site_names'] = site_name
    sites.at[site, 'site_cities'] = site_cities[sites.loc[site]['site_nums']]
    for region in regions.keys():
        if site_name in regions[region]:
            sites.at[site, 'region'] = region

#locs = gpd.tools.geocode(sites.site_cities, provider='nominatim', user_agent="bott")

locs = pd.DataFrame()
  
#for city in sites.site_cities:
#    temp = gpd.tools.geocode(city, provider='nominatim', user_agent='klb')
#    locs = pd.concat(
#        [
#            locs,
#            temp
#        ],
#        axis=0
#    )
from shapely import wkt
#locs = pd.concat([locs, sites], axis=1)
locs = pd.read_csv('site_geography_MRI.csv', index_col=0)
locs['geometry'] = locs['geometry'].apply(wkt.loads)

for i in locs.index:
    for source in sources:
        site = locs.loc[i]['site_nums']
        avg = df[df['site_id_l'] == site][source].mean()
        std = df[df['site_id_l'] == site][source].std()
        locs.at[i, f'{source}_mean'] = avg
        locs.at[i, f'{source}_sdev'] = std
        if site == 'site15':
            mri = 'SIEMENS'
        else:
            mri = df[df['site_id_l'] == site]['mri_info_manufacturer'].unique()[0]
        locs.at[i, 'MRI'] = mri

weight = {}
model = {}
perf = {}

date = '07_31_2024'

for source in sources.keys():
    
    perf[source] = pd.read_csv(
        join(
            PROJ_DIR,
            OUTP_DIR, 
            'base',
            f"nbs-predict_outcome-{source}_model_performance-{date}.tsv",
        ), 
        sep='\t',
        index_col=0,
        header=0,
        #dtype=float
    )
    perf[source] = pd.concat([perf[source], sites], axis=1)


performance_corrs = pd.DataFrame(
  dtype=float,
  columns=pd.MultiIndex.from_product([['mean', 'dev'],['r', 'p']]),
  index=pd.MultiIndex.from_product([sources.keys(), ['base', 'delta']])
)

for source in sources.keys():
  temp = perf[source]
  
  #temp.index = temp['site_nums']
  temp2 = pd.concat([locs, temp], axis=1)
  temp2 = gpd.GeoDataFrame(temp2, crs='ESRI:102003')
  # create map of all states except AK and HI in the main map axis
  #fig,ax = plt.subplots(ncols=3, figsize=(11,4))
  fig,ax = plt.subplots(figsize=(4,4))
  usa = usa.to_crs("ESRI:102003")
  ax0 = usa.plot(color='0.8', edgecolor='0.8', ax=ax#[0]
                 )
  ax0.axis('off')
  ax0.set_title(sources[source])
  temp2.plot(
      ax=ax0, 
      column='mse', 
      legend=True,
      cmap='RdYlGn_r',
      #norm=colors.CenteredNorm(),
      #markersize=temp2[f'{source}_mean'] ** 2 * 25,
      legend_kwds={
          "label": "MSE", 
          #"label": "$R^2$",
          "orientation": "horizontal", 
          "shrink": 0.75,
          "aspect": 10
          },
  )

  #ax1 = usa.plot(color='0.8', edgecolor='0.8', ax=ax[1])
  #ax1.axis('off')
  r,p = spearmanr(temp2['mse'], temp2[f'{source}_mean'], nan_policy='omit')
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
  r,p = spearmanr(temp2['mse'], temp2[f'{source}_sdev'], nan_policy='omit')
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


mega = pd.concat(
    [perf[i].melt(value_name=sources[i], var_name=i) for i in list(perf.keys())
    ], 
    axis=1
)
mega = mega.drop(['F3','F4', 'F5','F2', 'F6'
                  ], axis=1)
mega = mega.rename({'F1': 'variable'}, axis=1)



fig,ax = plt.subplots(figsize=(10,6))
g = sns.kdeplot(
    mega[mega['variable'] == 'corr'], 
    fill=True, 
    palette=source_pal, 
    ax=ax,
    legend=True
)
legend = g.get_legend()
legend.set_bbox_to_anchor((0.7,1))
ax.set_xlabel('Spearman Correlation: Actual vs. Predicted Exposure', fontsize=24)
#ax.set_xlim(-0.2,0.75)
sns.despine()
ax.axvline(0, lw=2, ls='--', color='#333333', alpha=0.4)
fig.savefig(f'figures/baseline-performance_spearmanr.png', dpi=600, bbox_inches='tight')


fig,ax = plt.subplots(figsize=(4,4))
g = sns.kdeplot(
    mega[mega['variable'] == 'mse'], 
    fill=True, 
    #alpha=0.1,
    palette=source_pal, 
    ax=ax,
    legend=True
)
ax.set_xlabel('Mean Squared Error', fontsize=24)
ax.set_xlim(0,6)
leg = ax.get_legend()#.remove()
leg.set_bbox_to_anchor((1,-.2))
sns.despine()
#ax.axvline(0, lw=2, ls='--', color='#333333', alpha=0.4)

fig.savefig(f'figures/baseline-performance_mse.png', dpi=600, bbox_inches='tight')


sources = {
    #'F1': 'Crustal', # no exposure-related FC
    #'F2': 'Ammonium Sulfates', # no exposure-related FC
    'F3': 'Biomass Burning',
    'F4': 'Traffic',
    'F5': 'Ammonium Nitrates',
    'F6': 'Industrial Fuel'
}

source_pal = sns.color_palette(
    [
        #'#2B9ADE', 
        #'#785EF0', 
        '#DC267F', 
        '#FE6100', 
        '#FFB000', 
        '#858686'
    ]
)


model = {}
perf = {}

date = '08_01_2024'

for source in sources.keys():
    i = list(sources.keys()).index(source)
    name = sources[source]
    perf[source] = pd.read_csv(
        join(
            PROJ_DIR,
            OUTP_DIR, 
            'delta',
            f"nbs-predict_outcome-{source}_model_performance-{date}.tsv",
        ), 
        sep='\t',
        index_col=0,
        header=0,
        #dtype=float
    )
    perf[source].index = [
        'site16',
        'site02',
        'site14',
        'site09',
        'site21',
        'site03',
        'site12',
        'site06',
        'site05',
        'site07',
        'site20',
        'site11',
        'site15',]

locs.index = locs['site_nums']



for source in sources.keys():
  temp = perf[source]
  
  #temp.index = temp['site_nums']
  temp2 = pd.concat([locs, temp], axis=1)
  # create map of all states except AK and HI in the main map axis
  #fig,ax = plt.subplots(ncols=3, figsize=(11,4))
  fig,ax = plt.subplots(figsize=(4,4))
  usa = usa.to_crs("ESRI:102003")
  ax0 = usa.plot(color='0.8', edgecolor='0.8', ax=ax#[0]
                 )
  ax0.axis('off')
  ax0.set_title(sources[source])
  temp2.to_crs("ESRI:102003").plot(
      ax=ax0, 
      column='mse', 
      legend=True,
      cmap='RdYlGn_r',
      #norm=colors.CenteredNorm(),
      markersize=temp2[f'{source}_mean'] ** 2 * 25,
      legend_kwds={
          "label": "MSE", 
          #"label": "$R^2$",
          "orientation": "horizontal", 
          "shrink": 0.75,
          "aspect": 10
          },
  )
  nans = temp2[temp2['corr'].isna()]
  nans.to_crs("ESRI:102003").plot(
      ax=ax0,  
      legend=True,
      #column=source,
      color='0.5',
      marker='x',
      markersize=nans[f'{source}_mean'] ** 2 * 25,
  )

  #ax1 = usa.plot(color='0.8', edgecolor='0.8', ax=ax[1])
  #ax1.axis('off')
  r,p = spearmanr(temp2['mse'], temp2[f'{source}_mean'], nan_policy='omit')
  performance_corrs.at[(source, 'delta'), ('mean','r')] = r
  performance_corrs.at[(source, 'delta'), ('mean','p')] = p
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
  r,p = spearmanr(temp2['mse'], temp2[f'{source}_sdev'], nan_policy='omit')
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

mega = pd.concat(
    [perf[i].melt(value_name=sources[i], var_name=i) for i in list(perf.keys())
    ], 
    axis=1
)
mega = mega.drop(['F3','F4', 'F5', 
                  ], axis=1)
mega = mega.rename({'F6': 'variable'}, axis=1)

spearmanr(temp2['mse'], temp2[f'{source}_sdev'], nan_policy='omit')

#sns.set_style('white')
fig,ax = plt.subplots(figsize=(10,6))
g = sns.kdeplot(
    mega[mega['variable'] == 'corr'], 
    fill=True, 
    palette=source_pal, 
    ax=ax,
    legend=True
)
legend = g.get_legend()
legend.set_bbox_to_anchor((0.7,1))
ax.set_xlabel('Spearman Correlation: Actual vs. Predicted Exposure', fontsize=24)
#ax.set_xlim(-0.2,0.75)
sns.despine()
ax.axvline(0, lw=2, ls='--', color='#333333', alpha=0.4)
fig.savefig(f'figures/delta-performance_spearmanr.png', dpi=600, bbox_inches='tight')


fig,ax = plt.subplots(figsize=(4,4))
g = sns.kdeplot(
    mega[mega['variable'] == 'mse'], 
    fill=True, 
    #alpha=0.1,
    palette=source_pal, 
    ax=ax,
    legend=True
)
ax.set_xlabel('Mean Squared Error', fontsize=24)
ax.set_xlim(0,1.5)
ax.get_legend().remove()
sns.despine()
#ax.axvline(0, lw=2, ls='--', color='#333333', alpha=0.4)

fig.savefig(f'figures/delta-performance_mse.png', dpi=600, bbox_inches='tight')

performance_corrs

performance_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'mse_x_exposure-correlations_by_site.csv'))
