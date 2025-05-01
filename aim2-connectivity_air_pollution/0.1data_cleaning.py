#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import abcdWrangler as abcdw

from os.path import join


ABCD_DIR = "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release5.1/abcd-data-release-5.1"

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim2"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

DEMO_VARS = [
    "reshist_addr1_proxrd",
    "reshist_addr1_popdensity",
    "reshist_addr1_urban_area",
    "nsc_p_ss_mean_3_items",
    "ehi_y_ss_scoreb",
    #'reshist_addr1_Lnight_exi',
    'race_ethnicity_c_bl',
    'household_income_4bins_bl',
    "site_id_l",
    'rel_family_id',
]

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data.pkl"))

df = df.sort_index()
df['screentime'] = ((5 * df['stq_y_ss_weekday']) + (2 * df['stq_y_ss_weekend'])) / 7

df['interview_date'] = pd.to_datetime(df['interview_date'], format="%m/%d/%Y")
site_15 = df[df['site_id_l'] == 'site15'].index.get_level_values(0).unique()

trim_df = df.drop(site_15, axis=0)
address = trim_df['reshist_addr1_urban_area'].dropna().index.get_level_values(0).unique()
# scanner manufacturer & pre-covid only matters for change scores
# so we only need the ppt IDs 
siemens = trim_df[trim_df['mri_info_manufacturer'] == 'SIEMENS'].index.get_level_values(0).unique()

interlopers = trim_df.loc[siemens][trim_df.loc[siemens]['mri_info_manufacturer'] != "SIEMENS"]["mri_info_manufacturer"].dropna().index.get_level_values(0).unique()

siemens = list(set(siemens) - set(interlopers))

# QC filtering - censoring all ppts whose 2-year follow-up visit was after covid
# bc a global pandemic is a pretty serious confounder LOL
pre_covid = trim_df[trim_df["interview_date"] < '2020-03-01'].xs('2_year_follow_up_y_arm_1', level=1).index.get_level_values(0).unique()
change_score_eligible = list(set(siemens) & set(pre_covid) & set(address))

# returns index of ppts who meet inclusion criteria
good_fmri = abcdw.fmri_qc(trim_df, ntpoints=750, motion_thresh=0.5)

good_fmri_base = [i[0] for i in good_fmri if i[1] == 'baseline_year_1_arm_1']
good_fmri_base = list(set(good_fmri_base) & set(address))
complete_base = trim_df.loc[good_fmri_base][DEMO_VARS].dropna().index.get_level_values(0).unique()

siemens = list(set(siemens) & set(complete_base))
pre_covid = list(set(pre_covid) & set(complete_base))
good_fmri_y2fu = [i[0] for i in good_fmri if i[1] == '2_year_follow_up_y_arm_1']

good_fmri_delta = list(set(good_fmri_y2fu) & set(pre_covid))
siemens_change = list(set(good_fmri_delta) & set(siemens))

all_ppts = df.index.get_level_values(0).unique()
not_site15 = trim_df.index.get_level_values(0).unique()
sample_size = pd.DataFrame(
    columns=[
        'keep', 
        'drop'
    ],
    index=[
        'ABCD Study',
        'Not site15',
        'Address',
        'fMRI QC base',
        'base complete',
        'SIEMENS base',
        'SIEMENS change',
        'Pre-COVID',
        'delta complete',
    ]
)


sample_size.at['ABCD Study', 'keep'] = len(all_ppts)
sample_size.at['ABCD Study', 'drop'] = 0

sample_size.at['Not site15', 'keep'] = len(not_site15)
sample_size.at['Not site15', 'drop'] = len(all_ppts) - len(not_site15)


sample_size.at['Address', 'keep'] = len(address)
sample_size.at['Address', 'drop'] = len(not_site15) - len(address)

# imaging quality control at baselien
sample_size.at['fMRI QC base', 'keep'] = len(good_fmri_base)
sample_size.at['fMRI QC base', 'drop'] = len(address) - len(good_fmri_base)
# complete case data baseline
sample_size.at['base complete', 'keep'] = len(complete_base)
sample_size.at['base complete', 'drop'] = len(good_fmri_base) - len(complete_base)

sample_size.at['SIEMENS', 'keep'] = len(siemens)
sample_size.at['SIEMENS', 'drop'] = len(complete_base) - len(siemens)

sample_size.at['Pre-COVID', 'keep'] = len(pre_covid)
sample_size.at['Pre-COVID', 'drop'] = len(siemens) - len(pre_covid)

sample_size.at['delta complete', 'keep'] = len(good_fmri_delta)
sample_size.at['delta complete', 'drop'] = len(pre_covid) - len(good_fmri_delta)
# complete case data for change scores
sample_size.at['SIEMENS change', 'keep'] = len(list(set(siemens) & set(good_fmri_delta)))
sample_size.at['SIEMENS change', 'drop'] = len(good_fmri_delta) - len(list(set(siemens) & set(good_fmri_delta)))

sample_size.to_csv(join(PROJ_DIR, OUTP_DIR, 'sample_size_qc.csv'))


col_to_df = {
    'ABCD Study': all_ppts,
    'Not site15': not_site15,
    'Address': address,
    'fMRI QC base': good_fmri_base,
    # both scanners, baseline data
    'base complete': list(set(complete_base)),
    # SIEMENS only, baseline data
    'SIEMENS base': siemens,
    'Pre-COVID': pre_covid,
    # SIEMENS only change scores
    'SIEMENS change': siemens_change,
    # both scanners, change scores
    'delta complete': good_fmri_delta
    }

ppts = pd.DataFrame(
    index=all_ppts,
    columns=[
        'ABCD Study',
        'Not site15',
        'Address',
        'fMRI QC base',
        'base complete',
        'SIEMENS base',
        'SIEMENS change',
        'Pre-COVID',
        'delta complete',
    ]
)

for ppt in all_ppts:
    for key in col_to_df.keys():
        if ppt in col_to_df[key]:
            ppts.at[ppt, key] = 1

ppts.to_pickle(join(PROJ_DIR, OUTP_DIR, 'ppts_qc.pkl'))

COMP_VAR = [
    "reshist_addr1_br",
    "reshist_addr1_ca",
    "reshist_addr1_cu",
    "reshist_addr1_ec",
    "reshist_addr1_fe",
    "reshist_addr1_k",
    "reshist_addr1_nh4",
    "reshist_addr1_ni",
    "reshist_addr1_no3",
    "reshist_addr1_oc",
    "reshist_addr1_pb",
    "reshist_addr1_si",
    "reshist_addr1_so4",
    "reshist_addr1_v",
    "reshist_addr1_zn"
]

components = pd.read_csv(
    '/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release5.0/core/linked-external-data/led_l_particulat.csv', 
    index_col=[0,1]
).filter(like='reshist_addr1')[COMP_VAR]

model_vars = [
    "mri_info_manufacturer",
    "rsfmri_meanmotion",
    "rsfmri_ntpoints",
    "physical_activity1_y",
    "stq_y_ss_weekday", 
    "stq_y_ss_weekend",
    "race_ethnicity", 
    "rel_family_id",
    "sex",
    "interview_age",
    "ehi_y_ss_scoreb",
    #"site_id_l",
    "interview_date",
    "reshist_addr1_proxrd",
    "reshist_addr1_popdensity",
    "reshist_addr1_urban_area",
    "nsc_p_ss_mean_3_items",
    "reshist_addr1_pm25",
    "F1", 
    "F2", 
    "F3", 
    "F4", 
    "F5", 
    "F6",
    "rsfmri_meanmotion",
    "rsfmri_ntpoints",
    "physical_activity1_y",
    "stq_y_ss_weekday", 
    "stq_y_ss_weekend",
    ]
# both scanners
base_df = df.loc[complete_base].xs('baseline_year_1_arm_1', level=1)
base_df.to_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_base.pkl"))
# siemens only 
base_df = df.loc[siemens].xs('baseline_year_1_arm_1', level=1)
base_df.to_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_base-siemens.pkl"))

# both scanners
delta_df = df.loc[good_fmri_delta]
delta_df.to_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_delta.pkl"))
# siemens only
delta_df = df.loc[siemens_change]
delta_df.to_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_delta-siemens.pkl"))

table = pd.DataFrame(
    index=[
       "N"
    ], 
    columns=list(col_to_df.keys())
)

vars = [
    'interview_age',
    'demo_sex_v2_bl',
    'rsfmri_meanmotion',
    'race_ethnicity_c_bl',
    'household_income_4bins_bl',
    'reshist_addr1_popdensity',
    'mri_info_manufacturer',
    'ehi_y_ss_scoreb', 
    'reshist_addr1_proxrd',
    'reshist_addr1_urban_area', 
    'nsc_p_ss_mean_3_items',
    "reshist_addr1_pm252016aa",
    "physical_activity1_y",
    "stq_y_ss_weekday", 
    "stq_y_ss_weekend",
    "F1", 
    "F2", 
    "F3", 
    "F4", 
    "F5", 
    "F6"
] + COMP_VAR

df = pd.concat([df, components], axis=1)

for subset in col_to_df.keys():
    #print(subset, type(col_to_df[subset]))
    ppts = col_to_df[subset]
    temp_df = df.loc[ppts].xs('baseline_year_1_arm_1', level=1)
    table.at['N', subset] = len(temp_df.index)
    
    for col in vars:
        table.at[f'{col}-missing', subset] = temp_df[col].isna().sum()
        if len(temp_df[col].unique()) < 6:
            counts = temp_df[col].value_counts()
            for level in counts.index:
                table.at[f'{col}-{level}',subset] = counts[level]
        else:
            table.at[f'{col}-mean',subset] = temp_df[col].mean()
            table.at[f'{col}-sdev',subset] = temp_df[col].std()

table.dropna(how='all').to_csv(join(PROJ_DIR, OUTP_DIR, 'demographics.csv'))

sources = [
    'F1',
    'F2',
    'F3',
    'F4',
    'F5',
    'F6'
]

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


source_by_site = pd.DataFrame(
    dtype=str,
    index=trim_df['site_id_l'].dropna().unique(),
    columns=['N'] + sources
).drop('site22')

for site in source_by_site.index:
    temp = trim_df.loc[complete_base]
    temp = trim_df[trim_df['site_id_l'] == site].xs('baseline_year_1_arm_1', level=1)
    source_by_site.at[site, 'N'] = len(temp.index)
    for source in sources:
        mean = np.round(temp[source].mean(), 2)
        sdev = np.round(temp[source].std(), 2)
        source_by_site.at[site, source] = f'{mean} ± {sdev}'
source_by_site.rename(geo_order, axis=0).loc[geo_order.values()].to_csv(
    join(PROJ_DIR, OUTP_DIR, 'sources_by_site.csv')
)

nano = [
    "reshist_addr1_br",
    "reshist_addr1_ca",
    "reshist_addr1_cu",
    "reshist_addr1_fe",
    "reshist_addr1_k",
    "reshist_addr1_ni",
    "reshist_addr1_pb",
    "reshist_addr1_si",
    "reshist_addr1_v",
    "reshist_addr1_zn"
]
micro = list(set(COMP_VAR) - set(nano))

components.columns = [i.split('_')[-1] for i in components.columns]
nano = [i.split('_')[-1] for i in nano]
micro = [i.split('_')[-1] for i in micro]

base = components.loc[complete_base]
delta = components.loc[siemens_change]

base['Sample'] = 'Cross-sectional'
delta['Sample'] = 'Longitudinal'

ap_nano = pd.concat(
    [
        base.reset_index()[nano + ['Sample']],
        delta.reset_index()[nano + ['Sample']]
    ],
    axis=0
).melt(id_vars='Sample')

ap_micro = pd.concat(
    [
        base.reset_index()[micro + ['Sample']],
        delta.reset_index()[micro + ['Sample']]
    ],
    axis=0
).melt(id_vars='Sample')

sns.set(context='paper', style='ticks', font_scale=1.2)

font_path = '/Users/katherine.b/Library/Fonts/Raleway-Regular.ttf'  # Your font path goes here
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

fig,ax = plt.subplots(
    ncols=2, 
    figsize=(10,3),
    gridspec_kw={'width_ratios': [2, 1]},
    #layout='constrained'
)
g = sns.boxenplot(
    ap_nano, 
    x='variable', 
    y='value', 
    hue='Sample', 
    ax=ax[0]
)
g.set_xticklabels([i.capitalize() for i in nano])
g.set_xlabel('')
g.set_ylabel('$ng/m^3$')
g.set_yscale('log')
g.get_legend().remove()
h = sns.boxenplot(
    ap_micro, 
    x='variable', 
    y='value', 
    hue='Sample', 
    ax=ax[1]
)
h.set_xticklabels(
    [
        "$NH_4^+$",
        "$NO_3^-$",
        "$OC$",
        "$EC$",
        "$SO_4^2-$"
    ]
)
h.set_ylabel('$\mu g/m^3$')
h.set_xlabel('')
h.legend(ncols=2, bbox_to_anchor=(0, -0.15))
plt.subplots_adjust(wspace=0.25)
sns.despine()

fig.savefig(
    join(PROJ_DIR, FIGS_DIR, 'PM_component_plot.png'),
    dpi=600,
    facecolor='#FFFFFF',
    bbox_inches='tight'
)