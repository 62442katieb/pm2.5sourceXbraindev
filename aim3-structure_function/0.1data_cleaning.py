#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import abcdWrangler as abcdw

from os.path import join

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim3"
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

df['interview_date'] = pd.to_datetime(df['interview_date'], format="%m/%d/%Y")

trim_df = df[df['site_id_l'] != 'site15']
address = trim_df['reshist_addr1_urban_area'].dropna().index.get_level_values(0).unique()
# scanner manufacturer & pre-covid only matters for change scores
# so we only need the ppt IDs 
siemens = trim_df[trim_df['mri_info_manufacturer'] == 'SIEMENS'].index.get_level_values(0).unique()

interlopers = trim_df.loc[siemens][trim_df.loc[siemens]['mri_info_manufacturer'] != "SIEMENS"]["mri_info_manufacturer"].dropna().index.get_level_values(0).unique()

siemens = list(set(siemens) - set(interlopers))

# QC filtering - censoring all ppts whose 2-year follow-up visit was after covid
# bc a global pandemic is a pretty serious confounder LOL
pre_covid = trim_df[trim_df["interview_date"] < '2020-03-01'].xs('2_year_follow_up_y_arm_1', level=1).index.get_level_values(0).unique()

# returns index of ppts who meet inclusion criteria
good_fmri = abcdw.fmri_qc(trim_df, ntpoints=750, motion_thresh=0.5)
good_fmri_base = [i[0] for i in good_fmri if i[1] == 'baseline_year_1_arm_1']
#good_fmri_base = list(set(good_fmri_base) & set(address))

good_dmri = abcdw.dmri_qc(trim_df, motion_thresh=2)
good_dmri_base = [i[0] for i in good_dmri if i[1] == 'baseline_year_1_arm_1']
#good_dmri_base = list(set(good_dmri_base) & set(address))

good_mri_base = list(set(good_fmri_base) & set(good_dmri_base))
complete_base = df.loc[good_mri_base][DEMO_VARS].dropna().index.get_level_values(0).unique()

siemens = list(set(siemens) & set(complete_base))
pre_covid = list(set(pre_covid) & set(siemens))
good_fmri_y2fu = [i[0] for i in good_fmri if i[1] == '2_year_follow_up_y_arm_1']
good_dmri_y2fu = [i[0] for i in good_dmri if i[1] == '2_year_follow_up_y_arm_1']

good_fmri_delta = list(set(good_fmri_y2fu) & set(pre_covid))
good_mri_delta = list(set(good_fmri_delta) & set(good_dmri_y2fu))

base_w_address = list(set(complete_base) & set(address))
delta_w_address = list(set(good_mri_delta) & set(address))


all_ppts = df.index.get_level_values(0).unique()
sample_size = pd.DataFrame(
    columns=[
        'keep', 
        'drop'
    ],
    index=[
        'ABCD Study',
        'Address',
        'fMRI QC base',
        'base complete',
        'base + AP',
        'SIEMENS',
        'Pre-COVID',
        'delta complete',
        'delta + AP'
    ]
)


sample_size.at['ABCD Study', 'keep'] = len(all_ppts)
sample_size.at['ABCD Study', 'drop'] = 0


sample_size.at['Not site15', 'keep'] = len(trim_df.index)
sample_size.at['Not site15', 'drop'] = len(all_ppts) - len(trim_df.index)

sample_size.at['Address', 'keep'] = len(address)
sample_size.at['Address', 'drop'] = len(all_ppts) - len(address)

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

sample_size.to_csv(join(PROJ_DIR, OUTP_DIR, 'sample_size_qc.csv'))


col_to_df = {
    'ABCD Study': all_ppts,
    'Address': address,
    'MRI QC base': good_mri_base,
    'base complete': list(set(complete_base)),
    'base + AP': base_w_address,
    'SIEMENS': siemens,
    'Pre-COVID': pre_covid,
    'delta complete': good_mri_delta,
    'delta + AP': delta_w_address
    }

ppts = pd.DataFrame(
    index=all_ppts,
    columns=[
        'ABCD Study',
        'Address',
        'fMRI QC base',
        'base complete',
        'SIEMENS',
        'Pre-COVID',
        'delta complete',
    ]
)

for ppt in all_ppts:
    for key in col_to_df.keys():
        if ppt in col_to_df[key]:
            ppts.at[ppt, key] = 1

ppts.to_pickle(join(PROJ_DIR, OUTP_DIR, 'ppts_qc.pkl'))

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

base_df = trim_df.loc[complete_base].xs('baseline_year_1_arm_1', level=1)
base_df.to_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_base.pkl"))

delta_df = trim_df.loc[good_mri_delta]
delta_df.to_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_delta.pkl"))

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
]


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