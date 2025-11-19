import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import pandas as pd
from utilities.data_simulation.Download_data import download_data
import json
from src.wrappers.OsipiBase import OsipiBase
import matlab.engine

def load_data(anatomy):
    """Load b-values, b-vectors, and NIfTI data for the given anatomy."""
    base_path = '../../download/Data/'
    bval = np.genfromtxt(f'{base_path}{anatomy}.bval')
    nii = nib.load(f'{base_path}{anatomy}.nii.gz')
    data = nii.get_fdata()
    return bval, data, nii


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


def preprocess_data(data, bval, mask=None, slice_idx=None):
    """Preprocess the data by normalizing and filtering background noise. If slice_idx is given, process only one slice."""
    if slice_idx is not None:
        data = data[:, :, slice_idx, :]
        if mask is not None:
            mask = mask[:, :, slice_idx]

    sx, sy, sz, n_bval = data.shape if slice_idx is None else (*data.shape[:2], 1, data.shape[2])
    X_dw = np.reshape(data, (sx * sy * sz, n_bval))
    selsb = np.array(bval) == 0
    S0 = np.nanmean(X_dw[:, selsb], axis=1)
    S0[np.isnan(S0)] = 0
    valid_id = S0 > (0.5 * np.median(S0[S0 > 0]))

    if mask is not None:
        mask_flat = mask.flatten() > 0
        valid_id = valid_id & mask_flat

    data_not_norm = X_dw[valid_id, :]
    data_norm = data_not_norm / data_not_norm[:, 0][:, np.newaxis]
    return data_norm, valid_id, sx, sy, sz



def run_algorithms(algorithm_name, requires_matlab, deep_learning, data_norm, bval, valid_id, sx, sy, sz, anatomy, base_dir, nii,  slice_idx=None, eng=None):
    """Run multiple IVIM algorithms and save results to CSV."""

    if requires_matlab and eng is None:
        print(f"Skipping {algorithm_name}: requires Matlab")
        return

    print(algorithm_name)
    fit = OsipiBase(algorithm=algorithm_name,bvalues=bval)
    maps = fit.osipi_fit(data_norm[np.newaxis, :], bval)

    f_array, Dstar_array, D_array = maps["f"], maps["Dp"], maps["D"]

    # Initialize full-size arrays
    f_map, Dstar_map, D_map = np.zeros(sx * sy * sz), np.zeros(sx * sy * sz), np.zeros(sx * sy * sz)
    f_map[valid_id], Dstar_map[valid_id], D_map[valid_id] = np.squeeze(f_array), np.squeeze(Dstar_array),np.squeeze(D_array)

    # Save results to CSV
    suffix = f'_slice{slice_idx}' if slice_idx is not None else ''
    df = pd.DataFrame({"f": f_map, "D*": Dstar_map, "D": D_map})
    df.to_csv(fr'{base_dir}/IVIM_{anatomy}_{name}{suffix}.csv', index=False)
    print(f'Saved results for {name} ({anatomy})')
    output_file_f = base_dir + rf"\{algorithm_name}_{anatomy}{suffix}_fit_f.nii"
    output_file_D = base_dir + rf"\{algorithm_name}_{anatomy}{suffix}_fit_D.nii"
    output_file_Dp = base_dir + rf"\{algorithm_name}_{anatomy}{suffix}_fit_Dp.nii"
    nib.save(nib.Nifti1Image(f_map.reshape(sx,sy,sz), nii.affine),
             output_file_f)
    nib.save(nib.Nifti1Image(D_map.reshape(sx,sy,sz), nii.affine),
             output_file_D)
    nib.save(nib.Nifti1Image(Dstar_map.reshape(sx,sy,sz), nii.affine),
             output_file_Dp)




if __name__ == "__main__":
    slice_idx = None  # Set to an integer (e.g., 30) to process only a single 2D slice
    base_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(base_dir, exist_ok=True)
    project_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
    json_path = os.path.join(project_root, "tests", "IVIMmodels", "unit_tests", "algorithms.json")
    with open(json_path, "r") as f:
        config = json.load(f)
    # Build algorithm list
    algorithmlist = []
    for name in config["algorithms"]:
        settings = config.get(name, {})
        requires_matlab = settings.get("requires_matlab", False)
        deep_learning = settings.get("deep_learning", False)
        algorithmlist.append((name, requires_matlab, deep_learning))
    eng = matlab.engine.start_matlab()
    for anatomy in ["brain"]:
        bval, data, nifti = load_data(anatomy)
        data_norm, valid_id, sx, sy, sz = preprocess_data(data, bval, slice_idx=slice_idx)
        for name, requires_matlab, deep_learning in algorithmlist:
            run_algorithms(name, requires_matlab, deep_learning, data_norm, bval, valid_id, sx, sy, sz, anatomy, base_dir, nifti, slice_idx, eng=eng)