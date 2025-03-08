import nibabel as nib
import numpy as np
import pandas as pd
import os
import itertools
from scipy.stats import zscore
import sys

# Define constants
if len(sys.argv) < 2:
    raise ValueError("Please provide the subject number (SUB) as a command-line argument.")
SUB = sys.argv[1]
print(SUB)
# SUB = '01'
base_dir = f"/Users/hazimiasad/Documents/Work/megan/data/collection/Study1/sub-{SUB}"  # Replace with your actual path
func_dirs = [os.path.join(base_dir, "func", f"ses-{i+1}") for i in range(3)]
pattern_dir = os.path.join(base_dir, "pattern")
bold_files = [os.path.join(func_dir, f"sub-{SUB}_task-nfb_bold_run-{j+1}.nii.gz") 
             for func_dir, j in itertools.product(func_dirs, range(20))]
mask_file = os.path.join(pattern_dir, "roivox_mask.nii.gz")

# Load mask
mask_img = nib.load(mask_file)
mask_data = mask_img.get_fdata()  # Shape: (x, y, z)

# Process BOLD files
all_roi_timeseries = []
for bold_file in bold_files:
    if not os.path.exists(bold_file):
        continue
        # raise FileNotFoundError(f"BOLD file not found: {bold_file}")
    bold_img = nib.load(bold_file)
    bold_data = bold_img.get_fdata()  # Shape: (x, y, z, timepoints)
    n_timepoints = bold_data.shape[-1]
    roi_voxels = np.where(mask_data > 0)
    bold_2d = bold_data.reshape(-1, n_timepoints)
    voxel_indices = np.ravel_multi_index(roi_voxels, bold_data.shape[:3])  # Convert 3D indices to 1D
    roi_timeseries = bold_2d[voxel_indices, :]
    all_roi_timeseries.append(roi_timeseries)

# Combine timeseries
all_roi_timeseries = np.hstack(all_roi_timeseries)

# Z-score normalization
all_roi_timeseries_zscore = zscore(all_roi_timeseries, axis=1)

# Save results
save_path = f"../results/raw_data/sub-{SUB}"
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.save(os.path.join(save_path, 'all_roi_timeseries_zscore.npy'), all_roi_timeseries_zscore)