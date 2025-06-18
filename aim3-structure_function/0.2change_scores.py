import pandas as pd
import numpy as np
import pingouin as pg

from os.path import join

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim3"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

def sem_unequal_variance(df):
    if len(df.columns) == 2:
        time1 = df.T.iloc[0]
        time2 = df.T.iloc[1]
    elif len(df.index) == 2:
        time1 = df.iloc[0]
        time2 = df.iloc[1]
    s0 = time1.std()
    s2 = time2.std()
    r = np.corrcoef(time1,time2)[0,1]
    sem = np.sqrt(((s0 * np.sqrt(1 - r)) ** 2) + ((s2 * np.sqrt(1 - r)) ** 2))
    return sem

def residualize(X, y=None, confounds=None):
    '''
    all inputs need to be arrays, not dataframes
    '''
    # residualize the outcome
    if confounds is not None:
        if y is not None:
            temp_y = np.reshape(y, (y.shape[0],))
            y = pg.linear_regression(confounds, temp_y)
            resid_y = y.residuals_

            # residualize features
            resid_X = np.zeros_like(X)
            # print(X.shape, resid_X.shape)
            for i in range(0, X.shape[1]):
                X_temp = X[:, i]
                # print(X_temp.shape)
                X_ = pg.linear_regression(confounds, X_temp)
                # print(X_.residuals_.shape)
                resid_X[:, i] = X_.residuals_.flatten()
            return resid_y, resid_X
        else:
            # residualize features
            resid_X = np.zeros_like(X)
            # print(X.shape, resid_X.shape)
            for i in range(0, X.shape[1]):
                X_temp = X[:, i]
                # print(X_temp.shape)
                X_ = pg.linear_regression(confounds, X_temp)
                # print(X_.residuals_.shape)
                resid_X[:, i] = X_.residuals_.flatten()
            return resid_X


df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_delta.pkl"))
nones = list(df.filter(regex='rsfmri_c_ngd_.*_ngd_n').columns) + list(df.filter(regex='rsfmri_c_ngd_n_.*').columns)
ppt_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'ppts_qc.pkl'))

df = df.drop(nones, axis=1)

base = df.xs('baseline_year_1_arm_1', level=1)
y2fu = df.xs('2_year_follow_up_y_arm_1', level=1)

conns = base.filter(like="rsfmri_c_ngd_").columns
dmri_vars = base.filter(like='dmri_rsirn').columns

mri_vars = list(conns) + list(dmri_vars)

ppts = y2fu[conns].dropna().index

base_rsfc_resid = residualize(base.loc[ppts][conns].values, confounds=base.loc[ppts]["rsfmri_meanmotion"].values)
y2fu_rsfc_resid = residualize(y2fu.loc[ppts][conns].values, confounds=y2fu.loc[ppts]["rsfmri_meanmotion"].values)

base_dmri_resid = residualize(base.loc[ppts][dmri_vars].values, confounds=base.loc[ppts]["dmri_meanmotion"].values)
y2fu_dmri_resid = residualize(y2fu.loc[ppts][dmri_vars].values, confounds=y2fu.loc[ppts]["dmri_meanmotion"].values)
#y4fu_resid = residualize(y4fu_df.drop("rsfmri_meanmotion", axis=1).values, confounds=y4fu_df["rsfmri_meanmotion"].values)

base_rsfc_resid = pd.DataFrame(base_rsfc_resid, index=ppts, columns=conns)
y2fu_rsfc_resid = pd.DataFrame(y2fu_rsfc_resid, index=ppts, columns=conns)

base_dmri_resid = pd.DataFrame(base_dmri_resid, index=ppts, columns=dmri_vars)
y2fu_dmri_resid = pd.DataFrame(y2fu_dmri_resid, index=ppts, columns=dmri_vars)

base_resid = pd.concat([base_rsfc_resid,  base_dmri_resid], axis=1)
y2fu_resid = pd.concat([y2fu_rsfc_resid,  y2fu_dmri_resid], axis=1)

rci = pd.DataFrame(index=ppts, columns=mri_vars, dtype=float)
sign_change = pd.DataFrame(index=ppts, columns=conns, dtype=float)

for var in mri_vars:
    temp = pd.concat([base_resid[var], y2fu_resid[var]], axis=1)
            
    sem = sem_unequal_variance(temp.dropna())
    abs_sem = sem_unequal_variance(abs(temp.dropna()))
    for i in ppts:
            age0 = base.loc[i, 'interview_age'] / 12.
            age2 = y2fu.loc[i, 'interview_age'] / 12.
            base_mri = base_resid.loc[i, var]
            y2fu_mri = y2fu_resid.loc[i, var]
            # @Jo add simple_change using np.tanh(base)
            rci.at[i,var] = ((y2fu_mri - base_mri) / sem) / (age2 - age0)
            #print(base * y2fu)
            if var in conns:
                if base_mri * y2fu_mri > 0:
                    if y2fu_mri > 0:
                        sign_change.at[i, var] = '+ to +'
                    else:
                        sign_change.at[i, var] = '- to -'
                else:
                    if y2fu_mri > 0:
                        sign_change.at[i, var] = '- to +'
                    else:
                        sign_change.at[i, var] = '+ to -'

not_mri = list(set(df.columns) - set(mri_vars))

rci = pd.concat([rci, df.xs('baseline_year_1_arm_1', level=1)[not_mri]], axis=1).dropna()
sign_change.to_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-sign_changes.pkl'))
rci.to_pickle(join(PROJ_DIR, DATA_DIR, 'dat_qcd-rci.pkl'))
rci.to_csv(join(PROJ_DIR, DATA_DIR, 'dat_qcd-rci.csv'))