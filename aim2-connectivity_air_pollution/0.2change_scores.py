import numpy as np
import pandas as pd
import pingouin as pg

from os.path import join

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim2"
DATA_DIR = "data"
OUTP_DIR = "output"
FIGS_DIR = "figures"

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

df = df.drop(nones, axis=1)

base_rsfc = df.xs('baseline_year_1_arm_1', level=1)
y2fu_rsfc = df.xs('2_year_follow_up_y_arm_1', level=1)

conns = base_rsfc.filter(like="rsfmri_c_ngd_").columns

ppts = list(set(base_rsfc[conns].dropna().index) & set(y2fu_rsfc[conns].dropna().index))

base_resid = residualize(base_rsfc.loc[ppts][conns].values, confounds=base_rsfc.loc[ppts]["rsfmri_meanmotion"].values)
y2fu_resid = residualize(y2fu_rsfc.loc[ppts][conns].values, confounds=y2fu_rsfc.loc[ppts]["rsfmri_meanmotion"].values)
#y4fu_resid = residualize(y4fu_df.drop("rsfmri_meanmotion", axis=1).values, confounds=y4fu_df["rsfmri_meanmotion"].values)

base_resid = pd.DataFrame(base_resid, index=ppts, columns=conns)
y2fu_resid = pd.DataFrame(y2fu_resid, index=ppts, columns=conns)

rci = pd.DataFrame(index=ppts, columns=conns, dtype=float)
rci_abs = pd.DataFrame(index=ppts, columns=conns, dtype=float)
sign_change = pd.DataFrame(index=ppts, columns=conns, dtype=float)

for conn in conns:
    temp = pd.concat([base_resid[conn], y2fu_resid[conn]], axis=1)
            
    sem = sem_unequal_variance(temp.dropna())
    abs_sem = sem_unequal_variance(abs(temp.dropna()))
    for i in ppts:
        if i not in base_rsfc.index or i not in y2fu_rsfc.index:
            pass
        else:
            age0 = base_rsfc.loc[i, 'interview_age'] / 12.
            age2 = y2fu_rsfc.loc[i, 'interview_age'] / 12.
            base = base_resid.loc[i, conn]
            y2fu = y2fu_resid.loc[i, conn]
            rci.at[i,conn] = ((y2fu - base) / sem) / (age2 - age0)
            rci_abs.at[i,conn] = ((np.abs(y2fu) - np.abs(base)) / abs_sem) / (age2 - age0)
            #print(base * y2fu)
            if base * y2fu > 0:
                if y2fu > 0:
                    sign_change.at[i, conn] = '+ to +'
                else:
                    sign_change.at[i, conn] = '- to -'
            else:
                if y2fu > 0:
                    sign_change.at[i, conn] = '- to +'
                else:
                    sign_change.at[i, conn] = '+ to -'

not_fc = list(set(df.columns) - set(rci.columns))

rci = pd.concat([rci, df.xs('baseline_year_1_arm_1', level=1)[not_fc]], axis=1)
rci_abs = pd.concat([rci_abs, df.xs('baseline_year_1_arm_1', level=1)[not_fc]], axis=1)

sign_change.to_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-sign_changes.pkl'))
rci.to_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-rci.pkl'))
rci_abs.to_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-rci_abs.pkl'))

rci.to_csv(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-rci.csv'))
rci_abs.to_csv(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-rci_abs.csv'))