import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import pandas as pd
from utilities.data_simulation.Download_data import download_data
import json
import matplotlib.pyplot as plt

def load_mask(anatomy):
    """Load the corresponding mask for the given anatomy."""
    mask_paths = {
        "brain_gray_matter": '../../download/Data/brain_mask_gray_matter.nii.gz',
        "brain_white_matter": '../../download/Data/brain_mask_white_matter.nii.gz',
        "abdomen": '../../download/Data/mask_abdomen_homogeneous.nii.gz'
    }
    masks = {}
    for key, path in mask_paths.items():
        if anatomy in key:
            masks[key] = nib.load(path).get_fdata()
    return masks

def load_data(algorithm, anatomy, slice_idx):
    """Load b-values, b-vectors, and NIfTI data for the given anatomy."""
    base_dir = os.path.join(os.path.dirname(__file__), 'results')
    suffix = f'_slice{slice_idx}' if slice_idx is not None else ''
    data = pd.read_csv(rf'{base_dir}/IVIM_{anatomy}_{algorithm}{suffix}.csv')
    return data

slice_idx = 10  # Set to an integer (e.g., 30) to process only a single 2D slice
anatomy = 'brain'
base_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(base_dir, exist_ok=True)
project_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
json_path = os.path.join(project_root, "tests", "IVIMmodels", "unit_tests", "algorithms.json")
with open(json_path, "r") as f:
    config = json.load(f)
masks = load_mask(anatomy)

# Build algorithm list
algorithmlist = []
for name in config["algorithms"]:
    settings = config.get(name, {})
    algorithmlist.append(name)
datasets = {}
for name in algorithmlist:
    data = load_data(name, anatomy, slice_idx)
    datasets[name] = data



# Get the first two algorithms
alg_names = list(datasets.keys())[:2]
alg1, alg2 = alg_names[0], alg_names[1]

D1 = np.array(datasets[alg1]["D"])
D2 = np.array(datasets[alg2]["D"])

mask = np.array(masks['brain_gray_matter']).astype(bool)
mask = mask[:,:,slice_idx]# convert to boolean
mask = mask.flatten()

# Apply mask
D1_masked = D1[mask]
D2_masked = D2[mask]
D1_masked = np.clip(D1_masked, 0,1)
D2_masked = np.clip(D2_masked, 0,1)
plt.figure(figsize=(6,6))
plt.scatter(D1_masked, D2_masked, alpha=0.6)
plt.xlabel(f"{alg1} D values")
plt.ylabel(f"{alg2} D values")
plt.title(f"D Scatter Plot ({np.sum(mask)} voxels, mask==1)")
plt.legend()
plt.grid(True)
plt.show()