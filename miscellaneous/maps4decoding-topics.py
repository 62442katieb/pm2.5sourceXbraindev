
import json
import pyreadr
import numpy as np
import pandas as pd
import nibabel as nib
import abcdWrangler as abcdw

import os
from pprint import pprint

from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.dataset import Dataset
from nimare.decode import discrete
from nimare.utils import get_resource_path


from os.path import join
from sklearn.metrics import jaccard_score

PROJ_DIR = '/Users/katherine.b/Library/CloudStorage/GoogleDrive-bottenho@usc.edu/.shortcut-targets-by-id/1xh4G1lP8EfcpjoT66SwxWXG4dlqB_Cqe/PLSC_AP_ThickAreaVolume_05 24/ns-decoding'
IN_DIR = 'boot_ratio_brain'
DECODE_DIR = 'decoded_clean'
RAW_DIR = 'decoding_raw'

out_dir = os.path.abspath("/Users/katherine.b/Dropbox/Projects/metas/m")
os.makedirs(out_dir, exist_ok=True)

ns_terms = pd.read_csv(
    '/Users/katherine.b/Dropbox/Projects/metas/ns-v-bm-decoding/characterized-ns-terms.csv',
    index_col=1,
    header=0
)
un_terms = ns_terms[ns_terms['type'] != 'Functional']
fun_terms = ns_terms[ns_terms['type'] == 'Functional']

files = fetch_neurosynth(
    data_dir=out_dir,
    version="7",
    overwrite=False,
    source="abstract",
    vocab="LDA100",
)
# Note that the files are saved to a new folder within "out_dir" named "neurosynth".
#pprint(files)
neurosynth_db = files[0]

neurosynth_dset = convert_neurosynth_to_dataset(
    coordinates_file=neurosynth_db["coordinates"],
    metadata_file=neurosynth_db["metadata"],
    annotations_files=neurosynth_db["features"],
)
neurosynth_dset.save(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))
#print(neurosynth_dset)

# I made a csv indicating sign flips 
# 1 for no flip, -1 for flip
swapsies = pd.read_csv(
    join(PROJ_DIR, 'sign_flips.csv'),
    index_col=[0,1],
    header=0
)
mega_df = pd.DataFrame(dtype=float)
for i in swapsies.index:
    print(i)
    ap = i[0]
    brain = i[1]
    abcd_var = f'smri_{brain.lower()}_cdk'
    # read in data
    df = pyreadr.read_r(join(PROJ_DIR, IN_DIR, f'{ap}_{brain}_boot.rds'))[None]
    df.columns = [f'{ap}_{brain}_{j[-1]}' for j in df.columns]
    df.index = [i[2:] for i in df.index]
    mega_df = pd.concat([mega_df, df], axis=1)
    if brain == 'Vol':
        df.index = [f'smri_{brain.lower()}_scs_{i}' for i in df.index]
    else:
        df.index = [f'smri_{brain.lower()}_cdk_{i}' for i in df.index]
    
    #regions = abcdw.assign_region_names(df)
    # xform each dimension into a nifto
    for dim in df.columns:
        if np.abs(df[dim]).max() > 2.5:
            dimension = f'Dimension {dim[-1]}'
            #temp = regions[[dim, 'long_region', 'hemisphere']]
            pos_sig = temp[temp[dim] > 2.5]
            neg_sig = temp[temp[dim] < -2.5]
            sig = pd.concat([pos_sig, neg_sig], axis=0)
            sig.to_csv(join(PROJ_DIR, IN_DIR, f'{ap}_{brain}_{dim[-1]}-region_names.csv'))
            scalar = int(swapsies.loc[i][dimension])
            decoded_df = pd.DataFrame(
                columns=['pReverse', 'zReverse', 'probReverse'],
                index = pd.MultiIndex.from_product([sig.index, ['terms']])
            )
            decoder = discrete.NeurosynthDecoder(correction=None)
            decoder.fit(neurosynth_dset)
            all_ids = list([])
            for roi in sig.index:
                br = sig.loc[roi][dim] * scalar
                temp = pd.Series(index=[roi], data = [1.])
                nifti = abcdw.series_2_nifti(temp)
                # find all studies reporting activation in that roi
                ids = neurosynth_dset.get_studies_by_mask(nifti)
                nifti = None
                all_ids += ids
                all_ids = list(np.unique(all_ids))
            temp_decoded = decoder.transform(ids=all_ids)
            #print(i, all_ids)
            decoder = None
            temp_decoded.index = [i.split('_')[-1] for i in temp_decoded.index]
            temp_decoded = temp_decoded.sort_values(by="probReverse", ascending=False).iloc[:25]
            temp_decoded.to_csv(join(PROJ_DIR, RAW_DIR, f'{ap}_{brain}_{dim[-1]}-topic.csv'))
            #for term in temp_decoded.index:
            #    if term in fun_terms.index:
            #        decoded_df.at[term, 'pReverse'] = temp_decoded.loc[term]['pReverse']
            #        decoded_df.at[term, 'zReverse'] = temp_decoded.loc[term]['zReverse']
            #        decoded_df.at[term, 'probReverse'] = temp_decoded.loc[term]['probReverse']
            decoded_df.dropna(how='all').to_csv(join(PROJ_DIR, DECODE_DIR, f'{ap}_{brain}_{dim[-1]}-topic.csv'))
            #nifti = nib.Nifti1Image(array, nifti.affine)
            #save
            #nifti.to_filename(join(PROJ_DIR, f'{ap}_{brain}_{dim[-1]}.nii.gz'))


js = {
    'surface area: component dim 1 vs source dim 1': jaccard_score(
        mega_df['PMcomp_Area_1'].dropna().between(-2.5, 2.5, inclusive='neither') == False,
        mega_df['S_Area_1'].dropna().between(-2.5, 2.5, inclusive='neither') == False
        ),
    'thickness source dim 1 vs area source dim 1': jaccard_score(
        mega_df['S_Thick_1'].dropna().between(-2.5, 2.5, inclusive='neither') == False, 
        mega_df['S_Area_1'].dropna().between(-2.5, 2.5, inclusive='neither') == False
        )
}

with open(join(PROJ_DIR, 'jaccard_index.json'), 'w') as fp:
    json.dump(js, fp)