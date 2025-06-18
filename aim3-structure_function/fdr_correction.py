import pandas as pd
import numpy as np
import seaborn as sns

import pyreadr
from os.path import join


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

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim3/"
OUTP_DIR = "output"

pm_factors = pd.read_excel(
    '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx', 
    skiprows=0, 
    index_col=0, 
    header=1
)

rni_lx = pyreadr.read_r(
    join(PROJ_DIR, OUTP_DIR,"baseline_lx_plsc_rnifmri (1).rds"),
    #@index_col=0,
    #header=0
)[None]


rni_keep = range(10)

rni_lx = pd.concat([rni_lx[rni_keep], pm_factors], axis=1).dropna()

alphas = pd.DataFrame(
    index=['rni', 'rnd'],
    columns=['lme', 'wqs']
)

alphas.at['rni', 'lme'] = jili_sidak_mc(rni_lx, 0.05)[0]
alphas.at['rni', 'wqs'] = jili_sidak_mc(rni_lx[rni_keep], 0.05)[0]

rnd_lx = pyreadr.read_r(
    join(PROJ_DIR, OUTP_DIR,"baseline_lx_plsc_rnd_fmri (1).rds"),
    #@index_col=0,
    #header=0
)[None]

rnd_keep = range(9)
rnd_lx = pd.concat([rnd_lx[rnd_keep], pm_factors], axis=1).dropna()
print('rnd, lx')
alphas.at['rnd', 'lme'] = jili_sidak_mc(rnd_lx, 0.05)[0]
alphas.at['rnd', 'wqs'] = jili_sidak_mc(rnd_lx[rnd_keep], 0.05)[0]

alphas.to_csv(join(PROJ_DIR, OUTP_DIR, 'plscXsource_regression-alphas.csv'))