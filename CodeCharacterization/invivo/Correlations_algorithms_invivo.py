import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import pandas as pd
from utilities.data_simulation.Download_data import download_data
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

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
r, p = pearsonr(D1_masked, D2_masked)

print(f"Pearson r = {r:.3f}, p = {p:.3e}")
plt.figure(figsize=(6,6))
plt.scatter(D1_masked, D2_masked, alpha=0.6)
m, b = np.polyfit(D1_masked, D2_masked, 1)
plt.plot(D1_masked, m*D1_masked + b, 'g-', label='Linear fit')
plt.xlabel(f"{alg1} D values")
plt.ylabel(f"{alg2} D values")
plt.title(f"D Scatter Plot ({np.sum(mask)} voxels, mask==1)")
plt.legend()
plt.grid(True)
plt.show()
algorithms = list(datasets.keys())
corr_matrix = pd.DataFrame(index=algorithms, columns=algorithms, dtype=float)

# Compute pairwise correlations
for alg1 in algorithms:
    D1 = np.array(datasets[alg1]["D"]).flatten()
    D1 = D1[mask]

    for alg2 in algorithms:
        D2 = np.array(datasets[alg2]["D"]).flatten()
        D2 = D2[mask]

        # Compute Pearson correlation
        r, _ = pearsonr(D1, D2)
        corr_matrix.loc[alg1, alg2] = r

        print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix.astype(float), annot=True, fmt=".2f",
            cmap="coolwarm", square=True, vmin=-1, vmax=1,
            cbar_kws={'label': 'Pearson r'})
plt.title(f"Algorithm Correlation Matrix (masked voxels: {np.sum(mask)})")
plt.show()

print("Pearson Correlation Matrix:")
print(corr_matrix.round(3))