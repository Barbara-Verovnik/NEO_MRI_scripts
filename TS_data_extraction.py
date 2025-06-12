#!/usr/bin/env python

#SBATCH --job-name=ts_atlas_64
#SBATCH --account=llonpp
#SBATCH --output=logs/output_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err
#SBATCH --nodes=1
#SBATCH --time 5:00:00
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --mem=45000M
#SBATCH --export=ALL

import os
print(os.environ['PYTHONPATH']) 

import runpy
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from nilearn import datasets
from nilearn._utils.niimg_conversions import check_same_fov
from nilearn.input_data import NiftiMapsMasker
from confounds import extract_confounds
import nibabel as nib

from tqdm import tqdm
from nilearn.image import load_img
from nilearn.maskers import NiftiMapsMasker
import math

from nilearn import connectome
from sklearn.covariance import LedoitWolf

dimension_of_atlas = 64

# Load list of preprocessed functional NIfTI files
with open("paths_to_nifti.pkl", "rb") as fp:
    func_imgs = pickle.load(fp)

path_to_excel = "demo_data_bin.xlsx"

df = pd.read_excel(path_to_excel)
ids = df["Subject ID how it's defined in lab/project"].to_numpy()

df_meta = pd.read_excel(path_to_excel)

# Load atlas and gray matter mask
fetcher = runpy.run_path('fetcher.py')
fetch_difumo = fetcher['fetch_difumo']
maps_img = fetch_difumo(dimension=dimension_of_atlas).maps

#downloads the mask to home/nilearn_data/icbm152_2009
gm_mask = datasets.fetch_icbm152_brain_gm_mask(threshold=0.2)


# === Define normalization function ===
def normalize_id(id_str):
    return id_str.split('_')[0]

def extract_subj_id_from_path(path):
    return os.path.basename(os.path.dirname(path))

# === Build mapping from normalized subject IDs to func paths ===
func_id_to_path = {}
for path in func_imgs:
    raw_id = extract_subj_id_from_path(path)
    norm_id = normalize_id(raw_id)
    func_id_to_path[norm_id] = path

# === Use 'ids' directly ===
meta_ids_norm = [normalize_id(str(mid)) for mid in ids]
common_ids = [mid for mid in meta_ids_norm if mid in func_id_to_path]
print(f"Subjects with both functional data and metadata (normalized): {len(common_ids)}")

aligned_func_paths = [func_id_to_path[mid] for mid in common_ids]
aligned_meta_ids = common_ids
assert len(aligned_func_paths) == len(aligned_meta_ids), "Mismatch after alignment."

# === Masker ===
mask_params = {
    'mask_img': gm_mask,
    'detrend': True,
    'standardize': True,
    'high_pass': 0.01,
    'low_pass': 0.1,
    't_r': 2.53,
    'smoothing_fwhm': 6.,
    'verbose': 1
}
masker = NiftiMapsMasker(maps_img=maps_img, **mask_params)

# Chunking parameters
n_chunks = 10
num_participants = len(aligned_func_paths)
ttotal = datetime.now()

for i, (func_img, subj_id) in enumerate(zip(aligned_func_paths, aligned_meta_ids)):
    print(f"\nProcessing subject {i+1}/{num_participants} ({subj_id})")
    t1 = datetime.now()

    img = nib.load(func_img)  # Load the NIfTI image
    confounds = extract_confounds(img, mask_img=gm_mask, n_confounds=10)
    signals = masker.fit_transform(img, confounds=confounds)

    T = signals.shape[0]
    chunk_len = T // n_chunks
    if chunk_len < 10:
        print(f"⚠️ Warning: Subject {subj_id} has very short time series ({T} TRs). Skipping.")
        continue

    all_chunks = []
    for j in range(n_chunks):
        start = j * chunk_len
        end = (j + 1) * chunk_len if j < n_chunks - 1 else T
        chunk_data = signals[start:end, :]
        all_chunks.append(chunk_data)

    # Save participant chunks to a separate file immediately
    out_path = f"chunked_data/64/{subj_id}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_chunks, f)

    # Clear variables to free memory (optional but recommended)
    del confounds, signals, all_chunks

    print(f"Saved chunks for {subj_id} to {out_path} (Elapsed: {datetime.now() - t1})")

print(f"\n🎉 All {num_participants} subjects processed in", str(datetime.now() - ttotal).split(".")[0])