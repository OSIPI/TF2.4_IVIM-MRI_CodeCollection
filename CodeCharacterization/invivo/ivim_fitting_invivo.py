import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import pandas as pd

from conftest import use_bounds, use_initial_guess
from utilities.data_simulation.Download_data import download_data
import json
from src.wrappers.OsipiBase import OsipiBase

def load_data(anatomy):
    """Load b-values, b-vectors, and NIfTI data for the given anatomy."""
    base_path = '/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/download/Data/'
    bval = np.genfromtxt(f'{base_path}{anatomy}.bval')
    nii = nib.load(f'{base_path}{anatomy}.nii.gz')
    data = nii.get_fdata()
    return bval, data, nii


def load_mask(anatomy):
    """Load the corresponding mask for the given anatomy."""
    mask_paths = {
        "brain_gray_matter": '/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/download/Data/Brain_mask_gray_matter.nii.gz',
        "brain_white_matter": '/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/download/Data/Brain_mask_white_matter.nii.gz',
        "abdomen": '/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/download/Data/mask_abdomen_homogeneous.nii.gz'
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



def run_algorithms(algorithm_name, requires_matlab, deep_learning, data_norm, bval, valid_id, sx, sy, sz, anatomy,
                   base_dir, nii, harmonization_string, slice_idx=None, eng=None, bounds=None, initial_guess=None):
    """Run multiple IVIM algorithms and save results to CSV."""

    if requires_matlab and eng is None:
        print(f"Skipping {algorithm_name}: requires Matlab")
        return
    print(algorithm_name)
    fit = OsipiBase(algorithm=algorithm_name, bvalues=bval, bounds=bounds, initial_guess=initial_guess)
    maps = fit.osipi_fit(data_norm, bval)

    f_array, Dstar_array, D_array = maps["f"], maps["Dp"], maps["D"]

    # Initialize full-size arrays
    f_map, Dstar_map, D_map = np.zeros(sx * sy * sz), np.zeros(sx * sy * sz), np.zeros(sx * sy * sz)
    f_map[valid_id], Dstar_map[valid_id], D_map[valid_id] = np.squeeze(f_array), np.squeeze(Dstar_array), np.squeeze(
        D_array)

    # Build rows matching the second script's CSV structure
    # Columns: ivim_algorithm, name, SNR, idx, f_true, Dp_true, D_true,
    #          f_fit, Dp_fit, D_fit,
    #          initial_guess_f, initial_guess_Dp, initial_guess_D,
    #          use_initial_guess_f, use_initial_guess_Dp, use_initial_guess_D,
    #          bounds_f, bounds_Dp, bounds_D,
    #          use_bounds_f, use_bounds_Dp, use_bounds_D,
    #          signal

    columns = [
        "ivim_algorithm", "name", "SNR", "idx",
        "f_true", "Dp_true", "D_true",
        "f_fit", "Dp_fit", "D_fit",
        "initial_guess_f", "initial_guess_Dp", "initial_guess_D",
        "use_initial_guess_f", "use_initial_guess_Dp", "use_initial_guess_D",
        "bounds_f", "bounds_Dp", "bounds_D",
        "use_bounds_f", "use_bounds_Dp", "use_bounds_D",
        "signal"
    ]

    if bounds is None:
        bounds = {'f': None, 'Dp': None, 'D': None}
    if initial_guess is None:
        initial_guess = {'f': None, 'Dp': None, 'D': None}

    rows = []
    for idx, (f_val, Dp_val, D_val) in enumerate(zip(f_map, Dstar_map, D_map)):
        rows.append([
            algorithm_name,  # ivim_algorithm
            name,  # name
            np.nan,  # SNR (unknown)
            idx,  # idx (voxel index)
            np.nan,  # f_true (unknown)
            np.nan,  # Dp_true (unknown)
            np.nan,  # D_true (unknown)
            f_val,  # f_fit
            Dp_val,  # Dp_fit
            D_val,  # D_fit
            initial_guess['f'],  # initial_guess_f
            initial_guess['Dp'],  # initial_guess_Dp
            initial_guess['D'],  # initial_guess_D
            fit.use_initial_guess['f'],  # use_initial_guess_f
            fit.use_initial_guess['Dp'],  # use_initial_guess_Dp
            fit.use_initial_guess['D'],  # use_initial_guess_D
            bounds['f'],  # bounds_f
            bounds['Dp'],  # bounds_Dp
            bounds['D'],  # bounds_D
            fit.use_bounds['f'],  # use_bounds_f
            fit.use_bounds['Dp'],  # use_bounds_Dp
            fit.use_bounds['D'],  # use_bounds_D
            np.nan  # signal (unknown)
        ])
    df = pd.DataFrame(rows, columns=columns)

    # Save results to CSV
    suffix = f'_slice{slice_idx}' if slice_idx is not None else ''
    csv_path = fr'{base_dir}/{harmonization_string}_{anatomy}{suffix}.csv'

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

    print(f'Saved results for {name} ({anatomy})')

    output_file_f = base_dir + rf"/{harmonization_string}_{algorithm_name}_{anatomy}{suffix}_fit_f.nii"
    output_file_D = base_dir + rf"/{harmonization_string}_{algorithm_name}_{anatomy}{suffix}_fit_D.nii"
    output_file_Dp = base_dir + rf"/{harmonization_string}_{algorithm_name}_{anatomy}{suffix}_fit_Dp.nii"
    nib.save(nib.Nifti1Image(f_map.reshape(sx,sy,sz), nii.affine),
             output_file_f)
    nib.save(nib.Nifti1Image(D_map.reshape(sx,sy,sz), nii.affine),
             output_file_D)
    nib.save(nib.Nifti1Image(Dstar_map.reshape(sx,sy,sz), nii.affine),
             output_file_Dp)


if __name__ == "__main__":
    slice_idx = 12
    use_bounds = True
    use_initial_guess = False

    base_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(base_dir, exist_ok=True)
    project_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
    json_path = os.path.join(project_root, "tests", "IVIMmodels", "unit_tests", "algorithms.json")
    with open(json_path, "r") as f:
        config = json.load(f)
    generic_path = os.path.join(project_root, "tests", "IVIMmodels", "unit_tests", "generic.json")
    with open(generic_path, "r") as f:
        generic = json.load(f)
    # Build algorithm list
    algorithmlist = []
    for name in config["algorithms"]:
        settings = config.get(name, {})
        requires_matlab = settings.get("requires_matlab", False)
        deep_learning = settings.get("deep_learning", False)
        algorithmlist.append((name, requires_matlab, deep_learning))
    for anatomy in ["Brain"]:
        bval, data, nifti = load_data(anatomy)
        if use_initial_guess:
            initial_guess = generic[anatomy].get("initial_guess", None)
        else:
            initial_guess = None

        if use_bounds:
            bounds = generic[anatomy].get("bounds", None)
        else:
            bounds = None
        data_norm, valid_id, sx, sy, sz = preprocess_data(data, bval, slice_idx=slice_idx)

        if not use_bounds and not use_initial_guess:
            harmonization_string = "no_harmonization"
        elif not use_bounds and use_initial_guess:
            harmonization_string = "initialguess_harmonized"
        elif use_bounds and not use_initial_guess:
            harmonization_string = "bounds_harmonized"
        elif use_bounds and use_initial_guess:
            harmonization_string = "bounds_and_initialguess_harmonized"

        for name, requires_matlab, deep_learning in algorithmlist:
            run_algorithms(name, requires_matlab, deep_learning, data_norm, bval, valid_id, sx, sy, sz, anatomy, base_dir, nifti, harmonization_string, slice_idx, bounds=bounds, initial_guess=initial_guess)