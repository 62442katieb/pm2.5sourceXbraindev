#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns

from os.path import join
from pyflowchart import *

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim3"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

qc_nums = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'sample_size_qc-baseline.csv')
)

st = StartNode("Full ABCD Study Sample\nN = 11,876")
s15 = ConditionNode("Site 15?")
ns15 = OperationNode("Not site 15\nN = 11,418")
ys15 = OperationNode("Exclude site 15\nN = 458")
dmriqc = ConditionNode("No/minimal incidental findings\n and dMRI FD < 2mm?")
guddmri = OperationNode("Quality dMRI data\nN = 8,966")
baddmri = OperationNode("Exclude poor quality dMRI data\nN = 2,452")
fmriqc = ConditionNode("10 minutes of low-motion rs-fMRI\nand mean FD < 0.5mm?")
badfmri = OperationNode("Exclude poor quality rs-fMRI data\nN = 2,633")
gudmri = OperationNode("Sufficient quality dMRI & rs-fMRI\nN = 6,333")
total = EndNode("Quality MRI and complete covariates\nN = 6320")

st.connect(s15)
s15.connect_no(ns15, "bottom")
s15.connect_yes(ys15, "right")
ns15.connect(dmriqc, "bottom")
dmriqc.connect_yes(guddmri, "bottom")
dmriqc.connect_no(baddmri, "right")
guddmri.connect(fmriqc, "bottom")
fmriqc.connect_yes(gudmri, "bottom")
fmriqc.connect_no(badfmri, "right")
gudmri.connect(total)

fc = Flowchart(st)
print(fc.flowchart())

output_html('mri_flowchart.html', "Full ABCD Study Sample\nN = 11,876", fc.flowchart())