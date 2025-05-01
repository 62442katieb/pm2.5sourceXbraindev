from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePGroupsOut, LeaveOneGroupOut, RepeatedKFold
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

import numpy as np
import scipy.io as sio
import pingouin as pg
import h5py

import time
from os.path import join
from datetime import datetime
from time import strftime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import warnings
import enlighten

warnings.simplefilter("ignore")


def residualize(X, confounds=None):
    '''
    all inputs need to be arrays, not dataframes
    '''
    # residualize the outcome
    resid_X = np.zeros_like(X)
    # print(X.shape, resid_X.shape)
    if len(X.shape) < 2:
        X_ = pg.linear_regression(confounds, X)
        # print(X_.residuals_.shape)
        resid_X = X_.residuals_.flatten()
    else:
        for i in range(0, X.shape[1]):
            X_temp = X[:, i]
            # print(X_temp.shape)
            X_ = pg.linear_regression(confounds, X_temp)
            # print(X_.residuals_.shape)
            resid_X[:, i] = X_.residuals_.flatten()
    return resid_X

def jili_sidak_mc(data, alpha):
    '''
    Accepts a dataframe (data, samples x features) and a type-i error rate (alpha, float), 
    then adjusts for the number of effective comparisons between variables
    in the dataframe based on the eigenvalues of their pairwise correlations.
    '''
    import math
    import numpy as np

    mc_corrmat = data.corr()
    mc_corrmat.fillna(0, inplace=True)
    eigvals, eigvecs = np.linalg.eig(mc_corrmat)

    M_eff = 0
    for eigval in eigvals:
        if abs(eigval) >= 0:
            if abs(eigval) >= 1:
                M_eff += 1
            else:
                M_eff += abs(eigval) - math.floor(abs(eigval))
        else:
            M_eff += 0
    print('\nFor {0} vars, number of effective comparisons: {1}\n'.format(mc_corrmat.shape[0], M_eff))

    #and now applying M_eff to the Sidak procedure
    sidak_p = 1 - (1 - alpha)**(1/M_eff)
    if sidak_p < 0.00001:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:2e} after corrections'.format(sidak_p))
    else:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:.6f} after corrections'.format(sidak_p))
    return sidak_p, M_eff

today = datetime.today()
today_str = strftime("%m_%d_%Y")

sns.set(context="paper", style='white', font_scale=1.2)


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim2"
#PROJ_DIR = "/home"
DATA_DIR = "data"
OUTP_DIR = "output"
FIGS_DIR = "figures"

OUTCOMES = [
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    ]
CONFOUNDS = ["demo_sex_v2_bl",
              "interview_age",
              "ehi_y_ss_scoreb",
              "site_id_l",
              "mri_info_manufacturer",
              "physical_activity1_y",
              "stq_y_ss_weekday",
              "stq_y_ss_weekend",
              "reshist_addr1_proxrd",
              "reshist_addr1_popdensity",
              "reshist_addr1_urban_area",
              "nsc_p_ss_mean_3_items",
              'race_ethnicity_c_bl',
              'household_income_4bins_bl',
              "rsfmri_meanmotion"
              ]
GROUPS = 'site_id_l'
TASK = "rest"
ATLAS = "gordon"
THRESH = 0.5
ALPHA = 0.05
K = 5
iter = 1000
atlas_fname = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/resources/gordon_networks_222.nii"

num_node = 12

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

as_factor = ["household_income_4bins_bl",
              "race_ethnicity_c_bl",
              "reshist_addr1_urban_area",
              "demo_sex_v2_bl",
              "mri_info_manufacturer",
              "site_id_l"]


dat = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_delta-siemens.pkl")).dropna()
#dat = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "delta_rsFC-rci_abs.pkl")).dropna()
print('rows =', len(dat.index), '\tcols =', len(dat.columns))
groups = dat[GROUPS]
edges = dat.filter(like='rsfmri_c_ngd')
rsfmri_vars = list(edges.columns)

if CONFOUNDS is not None:
    confounds = dat[CONFOUNDS]
else:
    confounds = None
# print(dat['bc'])
for confound in CONFOUNDS:
    if type(dat.iloc[0][confound]) != np.float64:
        print(confound, type(dat.iloc[0][confound]))
        temp = pd.get_dummies(dat[confound], dtype=int, prefix=confound)
        #print(temp.columns)
        confounds = pd.concat([confounds.drop(confound, axis=1), temp], axis=1)
    elif confound in as_factor:
        print(confound)
        temp = pd.get_dummies(dat[confound], dtype=int, prefix=confound)
        #print(temp.columns)
        confounds = pd.concat([confounds.drop(confound, axis=1), temp], axis=1)

linreg = LinearRegression()

#cv10 = GroupKFold(n_splits=10)
logo = LeaveOneGroupOut()
index = pd.MultiIndex.from_product(
    [
        OUTCOMES,
        groups.unique()
    ]
)

scores = pd.DataFrame(
    columns=edges.columns,
    index=index
)
corrs = pd.DataFrame(
    columns=edges.columns,
    index=index
)
r_sqs = pd.DataFrame(
    columns=edges.columns,
    index=index
)
manager = enlighten.get_manager()
tocks = manager.counter(
    total=len(index), 
    desc='Progress', 
    unit='models'
)

for outcome in OUTCOMES:
    for i, (train_index, test_index) in enumerate(logo.split(edges, dat[outcome], groups)):
        left_out = groups.iloc[test_index].unique()[0]
        #print(outcome, left_out)
        test_edges = edges.iloc[test_index]
        train_edges = edges.iloc[train_index]

        test_outcome = dat[outcome].iloc[test_index]
        train_outcome = dat[outcome].iloc[train_index]

        train_confounds = confounds.iloc[train_index]
        test_confounds = confounds.iloc[test_index]

        X_train = pd.DataFrame(
            residualize(train_edges.values, train_confounds.values),
            index=train_edges.index,
            columns=train_edges.columns
        )
        y_train = pd.Series(
            residualize(train_outcome.values, train_confounds.values),
            index=train_outcome.index,
            name=outcome
        )

        x = SelectFpr(f_regression, alpha=ALPHA).fit(X_train, y_train)
        sig_features = x.get_feature_names_out(X_train.columns)
        #print(len(sig_features))

        pos_features = []
        neg_features = []
        if len(sig_features) > 0:
            X_test = pd.DataFrame(
                residualize(test_edges[sig_features].values, test_confounds.values),
                index=test_edges.index,
                columns=sig_features
            )
            y_test = pd.Series(
                residualize(test_outcome.values, test_confounds.values),
                index=test_outcome.index,
                name=outcome
            )
            for feature in sig_features:
                corr = pd.concat([X_train[feature], y_train], axis=1).corr().loc[outcome, feature]
                corrs.at[(outcome, left_out), feature] = corr
                if corr > 0:
                    pos_features.append(feature)
                if corr < 0:
                    neg_features.append(feature)
            if len(pos_features) > 1:
                positive = X_train[pos_features].sum(axis=1)
                test_positive = X_test[pos_features].sum(axis=1)
            else:
                positive = X_train[pos_features]
                test_positive = X_test[pos_features]
            positive.name = 'positive'
            test_positive.name = 'positive'

            if len(neg_features) > 1:
                negative = X_train[neg_features].sum(axis=1)
                test_negative = X_test[neg_features].sum(axis=1)
            else:
                negative = X_train[neg_features]
                test_negative = X_test[neg_features]
            negative.name = 'negative'
            test_negative.name = 'negative'
            
            X_train = pd.concat([positive, negative], axis=1)
            X_test = pd.concat([test_positive, test_negative], axis=1)
            
            if len(X_test.columns) == 1:
                X_test = X_test.values.reshape(-1, 1)
            if len(X_train.columns) == 1:
                X_train = X_train.values.reshape(-1, 1)

            cve = linreg.fit(X_train, y_train)
            y_pred = cve.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rsq = r2_score(y_test, y_pred)
            for col in sig_features:
                scores.at[(outcome, left_out), col] = rmse
                r_sqs.at[(outcome, left_out), col] = rsq

        else:
            pass
        tocks.update()

scores.to_pickle(
    join(PROJ_DIR, OUTP_DIR, 'CPM-delta-mse.pkl')
)
scores.to_csv(
    join(PROJ_DIR, OUTP_DIR, 'CPM-delta-mse.csv')
)
corrs.to_pickle(
    join(PROJ_DIR, OUTP_DIR, 'CPM-delta-corrs.pkl')
)
corrs.to_csv(
    join(PROJ_DIR, OUTP_DIR, 'CPM-delta-corrs.csv')
)
