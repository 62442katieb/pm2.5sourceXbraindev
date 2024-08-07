# importing the tools we'll use throughout the rest of the script
# sys is system tools, should already be installed
import sys
# enlighten gives you a progress bar, but it's optional
import enlighten
# pandas is a dataframe-managing library and it's the absolute coolest
import pandas as pd
import numpy as np
import abcdWrangler as abcdw

# os is more system tools, should also already be installed
# we're importing tools for verifying and manipulating file paths/directories
from os.path import join, exists, isdir
from os import makedirs

DATA_DIR = (
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/"
)

everywhere_vars = ["collection_id", 
                   "abcd_adbc01_id", 
                   "dataset_id", 
                   "subjectkey", 
                   "src_subject_id", 
                   "interview_date", 
                   "interview_age", 
                   "sex", 
                   "eventname", 
                   "visit", 
                   "imgincl_t1w_include", 
                   "imgincl_t2w_include", 
                   "imgincl_dmri_include", 
                   "imgincl_rsfmri_include", 
                   "imgincl_mid_include", 
                   "imgincl_nback_include", 
                   "imgincl_sst_include"]

two_timept_vars = [
    "interview_date", 
    "imgincl_t1w_include", 
    "dmri_rsi_meanmotion", 
    "imgincl_dmri_include", 
    "rsfmri_c_ngd_ntpoints", 
    "mrif_score", 
    "imgincl_rsfmri_include", 
    "dmri_rsi_meanmotion"]

changes = ['abcd_smrip10201', 'abcd_smrip20201', 'abcd_smrip30201', 
           'abcd_mrisdp10201', 'abcd_mrisdp20201', 'abcd_dti_p101', 
           'abcd_drsip101', 'abcd_drsip201', 'abcd_mrirsfd01', 
           'abcd_mrirstv02', 'abcd_betnet02', 'mrirscor02', 'abcd_tbss01']

OUT_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim3/data"

# if the folder you want to save your dataset in doesn't exist, this will create it for you
if not isdir(OUT_DIR):
    makedirs(OUT_DIR)

# *exactly* as they appear in the ABCD data dictionary

variables = {
    "pdem02": [
        "demo_comb_income_v2",
        "demo_prnt_marital_v2",
        "demo_prnt_ed_v2",
    ],
    "abcd_yrb01": [
        "physical_activity1_y"
    ],
    "abcd_ssmty01": [
        "stq_y_ss_weekday", 
        "stq_y_ss_weekend"
    ],
    "acspsw03": [
        "race_ethnicity", 
        "rel_family_id",
        "sex",
        "interview_age"
    ],
    "abcd_ehis01": [
        "ehi_y_ss_scoreb"
    ],
    "abcd_lt01": [
        "site_id_l",
        "interview_date"
    ],
    "abcd_drsip101": [
        "dmri_rsi_meanmotion",
        "dmri_rsirnigm_cdk_bstslh", 
        "dmri_rsirnigm_cdk_caclh",
        "dmri_rsirnigm_cdk_cmflh", 
        "dmri_rsirnigm_cdk_cnlh", 
        "dmri_rsirnigm_cdk_erlh", 
        "dmri_rsirnigm_cdk_fflh", 
        "dmri_rsirnigm_cdk_iplh", 
        "dmri_rsirnigm_cdk_itlh", 
        "dmri_rsirnigm_cdk_iclh", 
        "dmri_rsirnigm_cdk_lolh", 
        "dmri_rsirnigm_cdk_loflh", 
        "dmri_rsirnigm_cdk_lglh", 
        "dmri_rsirnigm_cdk_moflh", 
        "dmri_rsirnigm_cdk_mtlh", 
        "dmri_rsirnigm_cdk_phlh", 
        "dmri_rsirnigm_cdk_pclh", 
        "dmri_rsirnigm_cdk_poplh", 
        "dmri_rsirnigm_cdk_poblh", 
        "dmri_rsirnigm_cdk_ptglh", 
        "dmri_rsirnigm_cdk_pcclh", 
        "dmri_rsirnigm_cdk_pctlh", 
        "dmri_rsirnigm_cdk_pcglh",
        "dmri_rsirnigm_cdk_prctlh",
        "dmri_rsirnigm_cdk_prcnlh",
        "dmri_rsirnigm_cdk_raclh", 
        "dmri_rsirnigm_cdk_rmflh", 
        "dmri_rsirnigm_cdk_sflh", 
        "dmri_rsirnigm_cdk_splh", 
        "dmri_rsirnigm_cdk_stlh", 
        "dmri_rsirnigm_cdk_smlh", 
        "dmri_rsirnigm_cdk_fplh", 
        "dmri_rsirnigm_cdk_tplh", 
        "dmri_rsirnigm_cdk_ttlh", 
        "dmri_rsirnigm_cdk_islh", 
        "dmri_rsirnigm_cdk_bstsrh", 
        "dmri_rsirnigm_cdk_cacrh", 
        "dmri_rsirnigm_cdk_cmfrh", 
        "dmri_rsirnigm_cdk_cnrh", 
        "dmri_rsirnigm_cdk_errh", 
        "dmri_rsirnigm_cdk_ffrh", 
        "dmri_rsirnigm_cdk_iprh", 
        "dmri_rsirnigm_cdk_itrh", 
        "dmri_rsirnigm_cdk_icrh", 
        "dmri_rsirnigm_cdk_lorh", 
        "dmri_rsirnigm_cdk_lofrh", 
        "dmri_rsirnigm_cdk_lgrh", 
        "dmri_rsirnigm_cdk_mofrh", 
        "dmri_rsirnigm_cdk_mtrh", 
        "dmri_rsirnigm_cdk_phrh", 
        "dmri_rsirnigm_cdk_pcrh", 
        "dmri_rsirnigm_cdk_poprh", 
        "dmri_rsirnigm_cdk_pobrh", 
        "dmri_rsirnigm_cdk_ptgrh",
        "dmri_rsirnigm_cdk_pccrh", 
        "dmri_rsirnigm_cdk_pctrh", 
        "dmri_rsirnigm_cdk_pcgrh", 
        "dmri_rsirnigm_cdk_prctrh", 
        "dmri_rsirnigm_cdk_prcnrh", 
        "dmri_rsirnigm_cdk_racrh", 
        "dmri_rsirnigm_cdk_rmfrh", 
        "dmri_rsirnigm_cdk_sfrh", 
        "dmri_rsirnigm_cdk_sprh",
        "dmri_rsirnigm_cdk_strh", 
        "dmri_rsirnigm_cdk_smrh", 
        "dmri_rsirnigm_cdk_fprh", 
        "dmri_rsirnigm_cdk_tprh", 
        "dmri_rsirnigm_cdk_ttrh", 
        "dmri_rsirnigm_cdk_isrh"
    ],
    "abcd_drsip201": [
        "dmri_rsirndgm_cdk_bstslh", 
        "dmri_rsirndgm_cdk_caclh",
        "dmri_rsirndgm_cdk_cmflh", 
        "dmri_rsirndgm_cdk_cnlh", 
        "dmri_rsirndgm_cdk_erlh", 
        "dmri_rsirndgm_cdk_fflh", 
        "dmri_rsirndgm_cdk_iplh", 
        "dmri_rsirndgm_cdk_itlh", 
        "dmri_rsirndgm_cdk_iclh", 
        "dmri_rsirndgm_cdk_lolh", 
        "dmri_rsirndgm_cdk_loflh", 
        "dmri_rsirndgm_cdk_lglh", 
        "dmri_rsirndgm_cdk_moflh", 
        "dmri_rsirndgm_cdk_mtlh", 
        "dmri_rsirndgm_cdk_phlh", 
        "dmri_rsirndgm_cdk_pclh", 
        "dmri_rsirndgm_cdk_poplh", 
        "dmri_rsirndgm_cdk_poblh", 
        "dmri_rsirndgm_cdk_ptglh", 
        "dmri_rsirndgm_cdk_pcclh", 
        "dmri_rsirndgm_cdk_pctlh", 
        "dmri_rsirndgm_cdk_pcglh",
        "dmri_rsirndgm_cdk_prctlh",
        "dmri_rsirndgm_cdk_prcnlh",
        "dmri_rsirndgm_cdk_raclh", 
        "dmri_rsirndgm_cdk_rmflh", 
        "dmri_rsirndgm_cdk_sflh", 
        "dmri_rsirndgm_cdk_splh", 
        "dmri_rsirndgm_cdk_stlh", 
        "dmri_rsirndgm_cdk_smlh", 
        "dmri_rsirndgm_cdk_fplh", 
        "dmri_rsirndgm_cdk_tplh", 
        "dmri_rsirndgm_cdk_ttlh", 
        "dmri_rsirndgm_cdk_islh", 
        "dmri_rsirndgm_cdk_bstsrh", 
        "dmri_rsirndgm_cdk_cacrh", 
        "dmri_rsirndgm_cdk_cmfrh", 
        "dmri_rsirndgm_cdk_cnrh", 
        "dmri_rsirndgm_cdk_errh", 
        "dmri_rsirndgm_cdk_ffrh", 
        "dmri_rsirndgm_cdk_iprh", 
        "dmri_rsirndgm_cdk_itrh", 
        "dmri_rsirndgm_cdk_icrh", 
        "dmri_rsirndgm_cdk_lorh", 
        "dmri_rsirndgm_cdk_lofrh", 
        "dmri_rsirndgm_cdk_lgrh", 
        "dmri_rsirndgm_cdk_mofrh", 
        "dmri_rsirndgm_cdk_mtrh", 
        "dmri_rsirndgm_cdk_phrh", 
        "dmri_rsirndgm_cdk_pcrh", 
        "dmri_rsirndgm_cdk_poprh", 
        "dmri_rsirndgm_cdk_pobrh", 
        "dmri_rsirndgm_cdk_ptgrh",
        "dmri_rsirndgm_cdk_pccrh", 
        "dmri_rsirndgm_cdk_pctrh", 
        "dmri_rsirndgm_cdk_pcgrh", 
        "dmri_rsirndgm_cdk_prctrh", 
        "dmri_rsirndgm_cdk_prcnrh", 
        "dmri_rsirndgm_cdk_racrh", 
        "dmri_rsirndgm_cdk_rmfrh", 
        "dmri_rsirndgm_cdk_sfrh", 
        "dmri_rsirndgm_cdk_sprh",
        "dmri_rsirndgm_cdk_strh", 
        "dmri_rsirndgm_cdk_smrh", 
        "dmri_rsirndgm_cdk_fprh", 
        "dmri_rsirndgm_cdk_tprh", 
        "dmri_rsirndgm_cdk_ttrh", 
        "dmri_rsirndgm_cdk_isrh"
    ],
    "abcd_betnet02": [
        "rsfmri_c_ngd_ntpoints",
        "rsfmri_c_ngd_ad_ngd_ad",
        "rsfmri_c_ngd_ad_ngd_cgc",
        "rsfmri_c_ngd_ad_ngd_ca",
        "rsfmri_c_ngd_ad_ngd_dt",
        "rsfmri_c_ngd_ad_ngd_dla",
        "rsfmri_c_ngd_ad_ngd_fo",
        "rsfmri_c_ngd_ad_ngd_n",
        "rsfmri_c_ngd_ad_ngd_rspltp",
        "rsfmri_c_ngd_ad_ngd_smh",
        "rsfmri_c_ngd_ad_ngd_smm",
        "rsfmri_c_ngd_ad_ngd_sa",
        "rsfmri_c_ngd_ad_ngd_vta",
        "rsfmri_c_ngd_ad_ngd_vs",
        "rsfmri_c_ngd_cgc_ngd_cgc",
        "rsfmri_c_ngd_cgc_ngd_ca",
        "rsfmri_c_ngd_cgc_ngd_dt",
        "rsfmri_c_ngd_cgc_ngd_dla",
        "rsfmri_c_ngd_cgc_ngd_fo",
        "rsfmri_c_ngd_cgc_ngd_n",
        "rsfmri_c_ngd_cgc_ngd_rspltp",
        "rsfmri_c_ngd_cgc_ngd_smh",
        "rsfmri_c_ngd_cgc_ngd_smm",
        "rsfmri_c_ngd_cgc_ngd_sa",
        "rsfmri_c_ngd_cgc_ngd_vta",
        "rsfmri_c_ngd_cgc_ngd_vs",
        "rsfmri_c_ngd_ca_ngd_ca",
        "rsfmri_c_ngd_ca_ngd_dt",
        "rsfmri_c_ngd_ca_ngd_dla",
        "rsfmri_c_ngd_ca_ngd_fo",
        "rsfmri_c_ngd_ca_ngd_n",
        "rsfmri_c_ngd_ca_ngd_rspltp",
        "rsfmri_c_ngd_ca_ngd_smh",
        "rsfmri_c_ngd_ca_ngd_smm",
        "rsfmri_c_ngd_ca_ngd_sa",
        "rsfmri_c_ngd_ca_ngd_vta",
        "rsfmri_c_ngd_ca_ngd_vs",
        "rsfmri_c_ngd_dt_ngd_dt",
        "rsfmri_c_ngd_dt_ngd_dla",
        "rsfmri_c_ngd_dt_ngd_fo",
        "rsfmri_c_ngd_dt_ngd_n",
        "rsfmri_c_ngd_dt_ngd_rspltp",
        "rsfmri_c_ngd_dt_ngd_smh",
        "rsfmri_c_ngd_dt_ngd_smm",
        "rsfmri_c_ngd_dt_ngd_sa",
        "rsfmri_c_ngd_dt_ngd_vta",
        "rsfmri_c_ngd_dt_ngd_vs",
        "rsfmri_c_ngd_dla_ngd_dla",
        "rsfmri_c_ngd_dla_ngd_fo",
        "rsfmri_c_ngd_dla_ngd_n",
        "rsfmri_c_ngd_dla_ngd_rspltp",
        "rsfmri_c_ngd_dla_ngd_smh",
        "rsfmri_c_ngd_dla_ngd_smm",
        "rsfmri_c_ngd_dla_ngd_sa",
        "rsfmri_c_ngd_dla_ngd_vta",
        "rsfmri_c_ngd_dla_ngd_vs",
        "rsfmri_c_ngd_fo_ngd_fo",
        "rsfmri_c_ngd_fo_ngd_n",
        "rsfmri_c_ngd_fo_ngd_rspltp",
        "rsfmri_c_ngd_fo_ngd_smh",
        "rsfmri_c_ngd_fo_ngd_smm",
        "rsfmri_c_ngd_fo_ngd_sa",
        "rsfmri_c_ngd_fo_ngd_vta",
        "rsfmri_c_ngd_fo_ngd_vs",
        "rsfmri_c_ngd_n_ngd_n",
        "rsfmri_c_ngd_n_ngd_rspltp",
        "rsfmri_c_ngd_n_ngd_smh",
        "rsfmri_c_ngd_n_ngd_smm",
        "rsfmri_c_ngd_n_ngd_sa",
        "rsfmri_c_ngd_n_ngd_vta",
        "rsfmri_c_ngd_n_ngd_vs",
        "rsfmri_c_ngd_rspltp_ngd_rspltp",
        "rsfmri_c_ngd_rspltp_ngd_smh",
        "rsfmri_c_ngd_rspltp_ngd_smm",
        "rsfmri_c_ngd_rspltp_ngd_sa",
        "rsfmri_c_ngd_rspltp_ngd_vta",
        "rsfmri_c_ngd_rspltp_ngd_vs",
        "rsfmri_c_ngd_smh_ngd_smh",
        "rsfmri_c_ngd_smh_ngd_smm",
        "rsfmri_c_ngd_smh_ngd_sa",
        "rsfmri_c_ngd_smh_ngd_vta",
        "rsfmri_c_ngd_smh_ngd_vs",
        "rsfmri_c_ngd_smm_ngd_smm",
        "rsfmri_c_ngd_smm_ngd_sa",
        "rsfmri_c_ngd_smm_ngd_vta",
        "rsfmri_c_ngd_smm_ngd_vs",
        "rsfmri_c_ngd_sa_ngd_sa",
        "rsfmri_c_ngd_sa_ngd_vta",
        "rsfmri_c_ngd_sa_ngd_vs",
        "rsfmri_c_ngd_vta_ngd_vta",
        "rsfmri_c_ngd_vta_ngd_vs",
        "rsfmri_c_ngd_vs_ngd_vs",
    ],
    "abcd_mri01": [
        "mri_info_manufacturer",
    ],
    "abcd_imgincl01": [
        "imgincl_t1w_include", 
        "imgincl_rsfmri_include",
        "imgincl_dmri_include"
    ],
    "abcd_mrfindings02": [
        "mrif_score"
    ],
    "abcd_rhds01": [
        "reshist_addr1_valid",
        "reshist_addr1_proxrd",
        "reshist_addr1_popdensity",
        "reshist_addr1_urban_area",
        "reshist_addr1_pm25",
        
    ],
    "abcd_sscep01": [
        "nsc_p_ss_mean_3_items"
    ]
}

timepoints = ["baseline_year_1_arm_1"]
change_scores = False

# reads in the data dictionary mapping variables to data structures
DATA_DICT = pd.read_csv(join(DATA_DIR, 'generate_dataset/data_element_names.csv'), index_col=0)

# read in csvs of interest one a time so you don't crash your computer
# grab the vars you want, then clear the rest and read in the next
# make one "missing" column for each modality if, like RSI, a subj is missing
# on all vals if missing on one. double check this.
# also include qa column per modality and make missingness chart before/after data censoring


# IF YOU WANT LONG FORMAT DATA, LONG=TRUE, IF YOU WANT WIDE FORMAT DATA, LONG=FALSE
long = True

# initialize the progress bars
manager = enlighten.get_manager()
tocks = manager.counter(total=len(variables.keys()), desc='Data Structures', unit='data structures')

# keep track of variables that don't make it into the big df
missing = {}

# build the mega_df now
df = pd.DataFrame()
data_dict = pd.DataFrame(columns=['data_structure', 'variable_description'])
for structure in variables.keys():
        
    missing[structure] = []
    old_columns = len(df.columns)
    path = join(DATA_DIR, 'csv', f'{structure}.csv')
    if exists(path):

        # original ABCD data structures are in long form, with eventname as a column
        # but I want the data in wide form, only one row per participant
        # and separate columns for values collected at different timepoints/events
        index = ["subjectkey", "eventname"]
        cols = variables[structure]
        
        if long == True:  
            if len(timepoints) > 1:
                temp_df = pd.read_csv(path, 
                              index_col=index, 
                              header=0, 
                              skiprows=[1], 
                              usecols= index + cols)
                temp1 = pd.DataFrame()
                for timepoint in timepoints:
                    print(timepoint)
                    temp2 = temp_df.xs(timepoint, level=1, drop_level=False)
                    temp1 = pd.concat([temp1, temp2], axis=0)
                temp_df = temp1
            else:
                temp_df0 = pd.read_csv(path, 
                              index_col="subjectkey", 
                              header=0, 
                              skiprows=[1], 
                              usecols= index + cols)
                temp_df = temp_df0[temp_df0['eventname'] == timepoints[0]]
                if structure in ["abcd_mrfindings02", "abcd_imgincl01", "abcd_betnet02", "abcd_lt01", "abcd_drsip101"]:
                    for variable in variables[structure]:
                        if variable in two_timept_vars:
                            temp_col = temp_df0[temp_df0['eventname'] == "2_year_follow_up_y_arm_1"][variable]
                            temp_col.name = f'{variable}2'
                            temp_df = pd.concat([temp_df, temp_col], axis=1)
                        else:
                            pass
                
            df = pd.concat([df, temp_df], axis=1)
            for variable in variables[structure]:
                try:
                    data_dict.at[variable, 'data_structure'] = structure
                    data_dict.at[variable, 'variable_description'] = DATA_DICT.loc[variable, 'description']
                except Exception as e:
                    print(e)
            
        else:
            #temp_df = pd.read_csv(path, index_col="subjectkey", header=0, skiprows=[1])
            if len(timepoints) > 1:
                temp_df = pd.read_csv(path, 
                              index_col=index, 
                              header=0, 
                              skiprows=[1], 
                              usecols= index + cols)
                temp1 = pd.DataFrame()
                for timepoint in timepoints:
                    temp2 = temp_df.xs(timepoint, level=1, drop_level=False)
                    temp_cols = [f'{col}.{timepoint}' for col in temp2.columns]
                    temp2.columns = temp_cols
                    temp1 = pd.concat([temp1, temp2], axis=1)
                temp_df = temp1
            else:
                temp_df = pd.read_csv(path, 
                              index_col="subjectkey", 
                              header=0, 
                              skiprows=[1], 
                              usecols= index + cols)
                temp_df = temp_df[temp_df['eventname'] == timepoints[0]]
            df = pd.concat([df, temp_df], axis=1)
        if change_scores:
            if structure in changes:
                path = join(DATA_DIR, 'change_scores', f'{structure}_changescores_bl_tp2.csv')
                change_cols = [f'{col}.change_score' for col in cols]
                index = ["subjectkey"]
                temp_df2 = pd.read_csv(path, 
                              index_col=index, 
                              header=0, 
                              usecols= index + change_cols)
                
                df = pd.concat([df, temp_df2], axis=1)
                
    new_columns = len(df.columns) - old_columns
    print(f"\t{new_columns} variables added!")
                
    if len(missing[structure]) >= 1:
        print(f"The following {len(missing[structure])}variables could not be added:\n{missing[structure]}")
    else:
        print(f"All variables were successfully added from {structure}.")
    temp_df = None
    tocks.update()


# how big is this thing?
print(f"Full dataframe is {sys.getsizeof(df) / 1000000}MB.")

df = df.dropna(how="all", axis=0)
df = df[df['site_id_l'] != 'site22']
df = df.loc[:,~df.columns.duplicated()].copy()

# let's grab all of the rsFC change scores
path = join(DATA_DIR, 'change_scores', 'abcd_betnet02_changescores_bl_tp2.csv')
change_scores = pd.read_csv(path, index_col="subjectkey", header=0)

# need a column for sign at baseline
for var in variables['abcd_betnet02']:
    base_change_scores = change_scores[f'{var}.baseline_year_1_arm_1'].copy()
    y2fu_change_scores = change_scores[f'{var}.2_year_follow_up_y_arm_1'].copy()
    change_scores[f'{var}.base_sign'] = np.sign(base_change_scores).copy()
    change_scores[f'{var}.2yfu_sign'] = np.sign(y2fu_change_scores).copy()
    abs_change = np.abs(y2fu_change_scores)  - np.abs(base_change_scores).copy()
    change_scores[f'{var}.change_sign'] = np.sign(abs_change).copy()

df = pd.concat([df, change_scores.filter(regex="rsfmri_c_.*", axis=1)], axis=1)

pm_factors = pd.read_excel('/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx', 
              skiprows=0, index_col=0, header=1)

noise = pd.read_csv('/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/preliminary_5.0/NPS_sound/Noise_12132021.csv', 
            index_col=0, header=0, usecols=["id_redcap", "reshist_addr1_Lnight_exi"])

data_dict.at['reshist_addr1_Lnight_exi', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/preliminary_5.0/NPS_sound/Noise_12132021.csv'
data_dict.at['reshist_addr1_Lnight_exi', 
             'variable_description'] = "average total sound level from the hours of 10p-7a at primary address"

data_dict.at['F1', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F1', 
             'variable_description'] = "Factor 1 from PMF of PM2.5, crustal materials - V, Si, Ca load highest"

data_dict.at['F2', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F2', 
             'variable_description'] = "Factor 2 from PMF of PM2.5, ammonium sulfates - SO4, NH4, V load highest"

data_dict.at['F3', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F3', 
             'variable_description'] = "Factor 3 from PMF of PM2.5, biomass burning - Br, K, OC load highest"

data_dict.at['F4', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F4', 
             'variable_description'] = "Factor 4 from PMF of PM2.5, traffic (TRAP) - Fe, Cu, EC load highest"

data_dict.at['F5', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F5', 
             'variable_description'] = "Factor 5 from PMF of PM2.5, ammonium nitrates - NH4, NO3, SO4 load highest"

data_dict.at['F6', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F6', 
             'variable_description'] = "Factor 6 from PMF of PM2.5, industrial fuel - Pb, Zn, Ni, Cu load highest"

data_dict.to_csv(join(OUT_DIR, 'data_dictionary.csv'))
big_df = pd.concat([df, pm_factors, noise], axis=1)
big_df.to_csv(join(OUT_DIR, 'data4.csv')
)


ABCD_DIR = "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release5.1/abcd-data-release-5.1"

variables = [
    "dmri_rsirnigm_cdk_bstslh", 
    "dmri_rsirnigm_cdk_caclh", 
    "dmri_rsirnigm_cdk_cmflh", 
    "dmri_rsirnigm_cdk_cnlh", 
    "dmri_rsirnigm_cdk_erlh", 
    "dmri_rsirnigm_cdk_fflh", 
    "dmri_rsirnigm_cdk_iplh", 
    "dmri_rsirnigm_cdk_itlh", 
    "dmri_rsirnigm_cdk_iclh", 
    "dmri_rsirnigm_cdk_lolh", 
    "dmri_rsirnigm_cdk_loflh", 
    "dmri_rsirnigm_cdk_lglh", 
    "dmri_rsirnigm_cdk_moflh", 
    "dmri_rsirnigm_cdk_mtlh", 
    "dmri_rsirnigm_cdk_phlh", 
    "dmri_rsirnigm_cdk_pclh", 
    "dmri_rsirnigm_cdk_poplh", 
    "dmri_rsirnigm_cdk_poblh", 
    "dmri_rsirnigm_cdk_ptglh", 
    "dmri_rsirnigm_cdk_pcclh", 
    "dmri_rsirnigm_cdk_pctlh", 
    "dmri_rsirnigm_cdk_pcglh", 
    "dmri_rsirnigm_cdk_prctlh", 
    "dmri_rsirnigm_cdk_prcnlh", 
    "dmri_rsirnigm_cdk_raclh", 
    "dmri_rsirnigm_cdk_rmflh", 
    "dmri_rsirnigm_cdk_sflh", 
    "dmri_rsirnigm_cdk_splh", 
    "dmri_rsirnigm_cdk_stlh", 
    "dmri_rsirnigm_cdk_smlh", 
    "dmri_rsirnigm_cdk_fplh", 
    "dmri_rsirnigm_cdk_tplh", 
    "dmri_rsirnigm_cdk_ttlh", 
    "dmri_rsirnigm_cdk_islh", 
    "dmri_rsirnigm_cdk_bstsrh", 
    "dmri_rsirnigm_cdk_cacrh", 
    "dmri_rsirnigm_cdk_cmfrh", 
    "dmri_rsirnigm_cdk_cnrh", 
    "dmri_rsirnigm_cdk_errh", 
    "dmri_rsirnigm_cdk_ffrh", 
    "dmri_rsirnigm_cdk_iprh", 
    "dmri_rsirnigm_cdk_itrh", 
    "dmri_rsirnigm_cdk_icrh", 
    "dmri_rsirnigm_cdk_lorh", 
    "dmri_rsirnigm_cdk_lofrh", 
    "dmri_rsirnigm_cdk_lgrh", 
    "dmri_rsirnigm_cdk_mofrh", 
    "dmri_rsirnigm_cdk_mtrh", 
    "dmri_rsirnigm_cdk_phrh", 
    "dmri_rsirnigm_cdk_pcrh", 
    "dmri_rsirnigm_cdk_poprh", 
    "dmri_rsirnigm_cdk_pobrh", 
    "dmri_rsirnigm_cdk_ptgrh", 
    "dmri_rsirnigm_cdk_pccrh", 
    "dmri_rsirnigm_cdk_pctrh", 
    "dmri_rsirnigm_cdk_pcgrh", 
    "dmri_rsirnigm_cdk_prctrh", 
    "dmri_rsirnigm_cdk_prcnrh", 
    "dmri_rsirnigm_cdk_racrh", 
    "dmri_rsirnigm_cdk_rmfrh", 
    "dmri_rsirnigm_cdk_sfrh", 
    "dmri_rsirnigm_cdk_sprh", 
    "dmri_rsirnigm_cdk_strh", 
    "dmri_rsirnigm_cdk_smrh", 
    "dmri_rsirnigm_cdk_fprh", 
    "dmri_rsirnigm_cdk_tprh", 
    "dmri_rsirnigm_cdk_ttrh", 
    "dmri_rsirnigm_cdk_isrh",
        "dmri_rsirni_scs_cbclh",
        "dmri_rsirni_scs_tplh",
        "dmri_rsirni_scs_cdlh",
        "dmri_rsirni_scs_ptlh",
        "dmri_rsirni_scs_pllh",
        "dmri_rsirni_scs_bs",
        "dmri_rsirni_scs_hclh",
        "dmri_rsirni_scs_aglh",
        "dmri_rsirni_scs_ablh",
        "dmri_rsirni_scs_vdclh",
        "dmri_rsirni_scs_cbcrh",
        "dmri_rsirni_scs_tprh",
        "dmri_rsirni_scs_cdrh",
        "dmri_rsirni_scs_ptrh",
        "dmri_rsirni_scs_plrh",
        "dmri_rsirni_scs_hcrh",
        "dmri_rsirni_scs_agrh",
        "dmri_rsirni_scs_abrh",
        "dmri_rsirni_scs_vdcrh",
    "dmri_rsirndgm_cdk_bstslh", 
    "dmri_rsirndgm_cdk_caclh", 
    "dmri_rsirndgm_cdk_cmflh", 
    "dmri_rsirndgm_cdk_cnlh", 
    "dmri_rsirndgm_cdk_erlh", 
    "dmri_rsirndgm_cdk_fflh", 
    "dmri_rsirndgm_cdk_iplh", 
    "dmri_rsirndgm_cdk_itlh", 
    "dmri_rsirndgm_cdk_iclh", 
    "dmri_rsirndgm_cdk_lolh", 
    "dmri_rsirndgm_cdk_loflh", 
    "dmri_rsirndgm_cdk_lglh", 
    "dmri_rsirndgm_cdk_moflh", 
    "dmri_rsirndgm_cdk_mtlh", 
    "dmri_rsirndgm_cdk_phlh", 
    "dmri_rsirndgm_cdk_pclh", 
    "dmri_rsirndgm_cdk_poplh", 
    "dmri_rsirndgm_cdk_poblh", 
    "dmri_rsirndgm_cdk_ptglh", 
    "dmri_rsirndgm_cdk_pcclh", 
    "dmri_rsirndgm_cdk_pctlh", 
    "dmri_rsirndgm_cdk_pcglh", 
    "dmri_rsirndgm_cdk_prctlh", 
    "dmri_rsirndgm_cdk_prcnlh", 
    "dmri_rsirndgm_cdk_raclh", 
    "dmri_rsirndgm_cdk_rmflh", 
    "dmri_rsirndgm_cdk_sflh", 
    "dmri_rsirndgm_cdk_splh", 
    "dmri_rsirndgm_cdk_stlh", 
    "dmri_rsirndgm_cdk_smlh", 
    "dmri_rsirndgm_cdk_fplh", 
    "dmri_rsirndgm_cdk_tplh", 
    "dmri_rsirndgm_cdk_ttlh", 
    "dmri_rsirndgm_cdk_islh", 
    "dmri_rsirndgm_cdk_bstsrh", 
    "dmri_rsirndgm_cdk_cacrh", 
    "dmri_rsirndgm_cdk_cmfrh", 
    "dmri_rsirndgm_cdk_cnrh", 
    "dmri_rsirndgm_cdk_errh", 
    "dmri_rsirndgm_cdk_ffrh", 
    "dmri_rsirndgm_cdk_iprh", 
    "dmri_rsirndgm_cdk_itrh", 
    "dmri_rsirndgm_cdk_icrh", 
    "dmri_rsirndgm_cdk_lorh", 
    "dmri_rsirndgm_cdk_lofrh", 
    "dmri_rsirndgm_cdk_lgrh", 
    "dmri_rsirndgm_cdk_mofrh", 
    "dmri_rsirndgm_cdk_mtrh", 
    "dmri_rsirndgm_cdk_phrh", 
    "dmri_rsirndgm_cdk_pcrh", 
    "dmri_rsirndgm_cdk_poprh", 
    "dmri_rsirndgm_cdk_pobrh", 
    "dmri_rsirndgm_cdk_ptgrh", 
    "dmri_rsirndgm_cdk_pccrh", 
    "dmri_rsirndgm_cdk_pctrh", 
    "dmri_rsirndgm_cdk_pcgrh", 
    "dmri_rsirndgm_cdk_prctrh", 
    "dmri_rsirndgm_cdk_prcnrh", 
    "dmri_rsirndgm_cdk_racrh", 
    "dmri_rsirndgm_cdk_rmfrh", 
    "dmri_rsirndgm_cdk_sfrh", 
    "dmri_rsirndgm_cdk_sprh", 
    "dmri_rsirndgm_cdk_strh", 
    "dmri_rsirndgm_cdk_smrh", 
    "dmri_rsirndgm_cdk_fprh", 
    "dmri_rsirndgm_cdk_tprh", 
    "dmri_rsirndgm_cdk_ttrh", 
    "dmri_rsirndgm_cdk_isrh",
        "dmri_rsirnd_scs_cbclh",
        "dmri_rsirnd_scs_tplh",
        "dmri_rsirnd_scs_cdlh",
        "dmri_rsirnd_scs_ptlh",
        "dmri_rsirnd_scs_pllh",
        "dmri_rsirnd_scs_bs",
        "dmri_rsirnd_scs_hclh",
        "dmri_rsirnd_scs_aglh",
        "dmri_rsirnd_scs_ablh",
        "dmri_rsirnd_scs_vdclh",
        "dmri_rsirnd_scs_cbcrh",
        "dmri_rsirnd_scs_tprh",
        "dmri_rsirnd_scs_cdrh",
        "dmri_rsirnd_scs_ptrh",
        "dmri_rsirnd_scs_plrh",
        "dmri_rsirnd_scs_hcrh",
        "dmri_rsirnd_scs_agrh",
        "dmri_rsirnd_scs_abrh",
        "dmri_rsirnd_scs_vdcrh",
    "rsfmri_var_cdk_banksstslh",
    "rsfmri_var_cdk_cdaclatelh",
    "rsfmri_var_cdk_cdmdflh",
    "rsfmri_var_cdk_cuneuslh",
    "rsfmri_var_cdk_entorhinallh",
    "rsfmri_var_cdk_fflh",
    "rsfmri_var_cdk_ifpalh",
    "rsfmri_var_cdk_iftlh",
    "rsfmri_var_cdk_ihclatelh",
    "rsfmri_var_cdk_loccipitallh",
    "rsfmri_var_cdk_loboflh",
    "rsfmri_var_cdk_linguallh",
    "rsfmri_var_cdk_moboflh",
    "rsfmri_var_cdk_mdtlh",
    "rsfmri_var_cdk_parahpallh",
    "rsfmri_var_cdk_paracentrallh",
    "rsfmri_var_cdk_parsopllh",
    "rsfmri_var_cdk_parsobalislh",
    "rsfmri_var_cdk_parstularislh",
    "rsfmri_var_cdk_pericclh",
    "rsfmri_var_cdk_postcentrallh",
    "rsfmri_var_cdk_psclatelh",
    "rsfmri_var_cdk_precentrallh",
    "rsfmri_var_cdk_precuneuslh",
    "rsfmri_var_cdk_rlaclatelh",
    "rsfmri_var_cdk_rlmdflh",
    "rsfmri_var_cdk_suflh",
    "rsfmri_var_cdk_spetallh",
    "rsfmri_var_cdk_sutlh",
    "rsfmri_var_cdk_smlh",
    "rsfmri_var_cdk_fpolelh",
    "rsfmri_var_cdk_tpolelh",
    "rsfmri_var_cdk_tvtlh",
    "rsfmri_var_cdk_insulalh",
    "rsfmri_var_cdk_banksstsrh",
    "rsfmri_var_cdk_cdaclaterh",
    "rsfmri_var_cdk_cdmdfrh",
    "rsfmri_var_cdk_cuneusrh",
    "rsfmri_var_cdk_entorhinalrh",
    "rsfmri_var_cdk_ffrh",
    "rsfmri_var_cdk_ifparh",
    "rsfmri_var_cdk_iftrh",
    "rsfmri_var_cdk_ihclaterh",
    "rsfmri_var_cdk_loccipitalrh",
    "rsfmri_var_cdk_lobofrh",
    "rsfmri_var_cdk_lingualrh",
    "rsfmri_var_cdk_mobofrh",
    "rsfmri_var_cdk_mdtrh",
    "rsfmri_var_cdk_parahpalrh",
    "rsfmri_var_cdk_paracentralrh",
    "rsfmri_var_cdk_parsoplrh",
    "rsfmri_var_cdk_parsobalisrh",
    "rsfmri_var_cdk_parstularisrh",
    "rsfmri_var_cdk_periccrh",
    "rsfmri_var_cdk_postcentralrh",
    "rsfmri_var_cdk_psclaterh",
    "rsfmri_var_cdk_precentralrh",
    "rsfmri_var_cdk_precuneusrh",
    "rsfmri_var_cdk_rlaclaterh",
    "rsfmri_var_cdk_rlmdfrh",
    "rsfmri_var_cdk_sufrh",
    "rsfmri_var_cdk_spetalrh",
    "rsfmri_var_cdk_sutrh",
    "rsfmri_var_cdk_smrh",
    "rsfmri_var_cdk_fpolerh",
    "rsfmri_var_cdk_tpolerh",
    "rsfmri_var_cdk_tvtrh",
    "rsfmri_var_cdk_insularh",
    "rsfmri_var_scs_crbcortexlh",
        "rsfmri_var_scs_tplh",
        "rsfmri_var_scs_caudatelh",
        "rsfmri_var_scs_putamenlh",
        "rsfmri_var_scs_pallidumlh",
        "rsfmri_var_scs_brainstem",
        "rsfmri_var_scs_hpuslh",
        "rsfmri_var_scs_amygdalalh",
        "rsfmri_var_scs_aalh",
        "rsfmri_var_scs_ventraldclh",
        "rsfmri_var_scs_crbcortexrh",
        "rsfmri_var_scs_tprh",
        "rsfmri_var_scs_caudaterh",
        "rsfmri_var_scs_putamenrh",
        "rsfmri_var_scs_pallidumrh",
        "rsfmri_var_scs_hpusrh",
        "rsfmri_var_scs_amygdalarh",
        "rsfmri_var_scs_aarh",
        "rsfmri_var_scs_ventraldcrh",
    "rsfmri_meanmotion",
    "rsfmri_ntpoints",
    "imgincl_rsfmri_include",
    "dmri_meanmotion",
    "imgincl_dmri_include",
    "mrif_score"
    
]

dat = abcdw.data_grabber(
    ABCD_DIR, 
    variables, 
    eventname='baseline_year_1_arm_1'
)

dat.to_pickle("/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim3/rsi_and_slo_data_baseline.pkl")
quality_rsfmri = abcdw.fmri_qc(dat, ntpoints=750, motion_thresh=0.5)
quality_dmri = abcdw.dmri_qc(dat, motion_thresh=2)

dat_qcd = pd.concat(
    [
        dat.filter(like='rsfmri').loc[quality_rsfmri], 
        dat.filter(like='dfmri').loc[quality_dmri]
    ], 
    axis=1
)

dat_qcd.to_csv("/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim3/data/rsi_and_slo_data_baseline-qcd.csv")