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
df = df.xs('baseline_year_1_arm_1', level=1)
all_ppts = df.index.get_level_values(0).unique()
trim_df = df[df['site_id_l'] != 'site15']

address = trim_df['reshist_addr1_urban_area'].dropna().index.get_level_values(0).unique()
# scanner manufacturer & pre-covid only matters for change scores
# so we only need the ppt IDs 

# returns index of ppts who meet inclusion criteria
good_fmri = abcdw.fmri_qc(trim_df, ntpoints=750, motion_thresh=0.5)
good_dmri = abcdw.dmri_qc(trim_df, motion_thresh=2)

good_mri = list(set(good_fmri) & set(good_dmri))
base_w_address = list(set(good_mri) & set(address))



sample_size = pd.DataFrame(
    columns=[
        'keep', 
        'drop'
    ],
    index=[
        'ABCD Study',
        'Not site15',
        'Address',
        'fMRI QC',
        'dMRI QC',
        'good mri',
        'mri + AP',
    ]
)

sample_size.at['Not site15', 'keep'] = len(trim_df.index)
sample_size.at['Not site15', 'drop'] = len(all_ppts) - len(trim_df.index)

sample_size.at['ABCD Study', 'keep'] = len(all_ppts)
sample_size.at['ABCD Study', 'drop'] = 0
sample_size.at['Address', 'keep'] = len(address)
sample_size.at['Address', 'drop'] = len(trim_df.index) - len(address)

# imaging quality control at baselien
sample_size.at['fMRI QC', 'keep'] = len(good_fmri)
sample_size.at['fMRI QC', 'drop'] = len(trim_df.index) - len(good_fmri)
sample_size.at['dMRI QC', 'keep'] = len(good_dmri)
sample_size.at['dMRI QC', 'drop'] = len(trim_df.index) - len(good_dmri)

sample_size.at['good mri', 'keep'] = len(good_mri)
sample_size.at['good mri', 'drop'] = len(trim_df.index) - len(good_mri)

sample_size.at['mri + AP', 'keep'] = len(base_w_address)
sample_size.at['mri + AP', 'drop'] = len(good_mri) - len(base_w_address)


# complete case data for change scores

sample_size.to_csv(join(PROJ_DIR, OUTP_DIR, 'sample_size_qc-baseline.csv'))


col_to_df = {
    'ABCD Study': all_ppts,
    'Address': address,
    'fMRI QC': good_fmri,
    'dMRI QC': good_dmri,
    'good mri': good_mri,
    'mri + AP': base_w_address,
}

ppts = pd.DataFrame(
    index=all_ppts,
    columns=col_to_df.keys()
)

for ppt in all_ppts:
    for key in col_to_df.keys():
        if ppt in col_to_df[key]:
            ppts.at[ppt, key] = 1

ppts.to_pickle(join(PROJ_DIR, OUTP_DIR, 'ppts_qc-baseline.pkl'))

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

base_df = trim_df.loc[good_mri]
base_df.to_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd-baseline.pkl"))
base_df.to_csv(join(PROJ_DIR, DATA_DIR, "data_qcd-baseline.csv"))

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
    temp_df = df.loc[ppts]
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

table.dropna(how='all').to_csv(join(PROJ_DIR, OUTP_DIR, 'demographics-baseline.csv'))