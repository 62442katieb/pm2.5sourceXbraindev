import pandas as pd
from os.path import join

PROJ_DIR = '/Users/katherine.b/Library/CloudStorage/GoogleDrive-bottenho@usc.edu/.shortcut-targets-by-id/1xh4G1lP8EfcpjoT66SwxWXG4dlqB_Cqe/PLSC_AP_ThickAreaVolume_05 24/ns-decoding'
IN_DIR = 'decoding_raw'
OUT_DIR = 'decoded_clean'


swapsies = pd.read_csv(
    join(PROJ_DIR, 'sign_flips.csv'),
    index_col=[0,1],
    header=0
)

redo = [
    ('S','Area'),
    ('PMcomp','Area')
]

ns_terms = pd.read_csv(
    '/Users/katherine.b/Dropbox/Projects/metas/ns-v-bm-decoding/characterized-ns-terms.csv',
    index_col=1,
    header=0
)

un_terms = ns_terms[ns_terms['type'] != 'Functional']
fun_terms = ns_terms[ns_terms['type'] == 'Functional']

#for i in swapsies.index:
for i in redo:
    print(i)
    ap = i[0]
    brain = i[1]
    df = pd.read_excel(join(PROJ_DIR, IN_DIR, f'{ap}_{brain}_1.xlsx'), index_col=0).dropna()
    for term in df.index:
        if term not in fun_terms.index:
            df = df.drop(term, axis=0)
    df.to_csv(join(PROJ_DIR, OUT_DIR, f'{ap}_{brain}-1.csv'))
    df = pd.read_excel(join(PROJ_DIR, IN_DIR, f'{ap}_{brain}_2.xlsx'), index_col=0).dropna()
    for term in df.index:
        if term not in fun_terms.index:
            df = df.drop(term, axis=0)
    df.to_csv(join(PROJ_DIR, OUT_DIR, f'{ap}_{brain}-2.csv'))