import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import pandas as pd
from utilities.data_simulation.Download_data import download_data
from src.standardized.ETP_SRI_LinearFitting import ETP_SRI_LinearFitting
from src.standardized.IAR_LU_biexp import IAR_LU_biexp
from src.standardized.IAR_LU_modified_mix import IAR_LU_modified_mix #--> gives error
from src.standardized.IAR_LU_modified_topopro import IAR_LU_modified_topopro #--> gives error
from src.standardized.IAR_LU_segmented_2step import IAR_LU_segmented_2step #--> gives error
from src.standardized.IAR_LU_segmented_3step import IAR_LU_segmented_3step #--> gives error
from src.standardized.IAR_LU_subtracted import IAR_LU_subtracted # --> gives error
from src.standardized.OGC_AmsterdamUMC_Bayesian_biexp import OGC_AmsterdamUMC_Bayesian_biexp
from src.standardized.OGC_AmsterdamUMC_biexp import OGC_AmsterdamUMC_biexp
from src.standardized.OGC_AmsterdamUMC_biexp_segmented import OGC_AmsterdamUMC_biexp_segmented
from src.standardized.OJ_GU_seg import OJ_GU_seg
from src.standardized.PvH_KB_NKI_IVIMfit import PvH_KB_NKI_IVIMfit
from src.standardized.PV_MUMC_biexp import PV_MUMC_biexp
from src.wrappers.OsipiBase import OsipiBase


def load_data(anatomy):
    """Load b-values, b-vectors, and NIfTI data for the given anatomy."""
    base_path = '../../download/Data/'
    bval = np.genfromtxt(f'{base_path}{anatomy}.bval')
    nii = nib.load(f'{base_path}{anatomy}.nii.gz')
    data = nii.get_fdata()
    return bval, data


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


def run_algorithms(data_norm, bval, valid_id, sx, sy, sz, anatomy, base_dir, mask_name, slice_idx=None):
    """Run multiple IVIM algorithms and save results to CSV."""
    algorithms = {
        "ETP_SRI_LinearFitting": ETP_SRI_LinearFitting,
        "OGC_AmsterdamUMC_Bayesian_biexp": OGC_AmsterdamUMC_Bayesian_biexp,
        "OGC_AmsterdamUMC_biexp": OGC_AmsterdamUMC_biexp,
        "OGC_AmsterdamUMC_biexp_segmented": OGC_AmsterdamUMC_biexp_segmented,
        "OJ_GU_seg": OJ_GU_seg,
        "PvH_KB_NKI_IVIMfit": PvH_KB_NKI_IVIMfit,
        "PV_MUMC_biexp": PV_MUMC_biexp,
        "IAR_LU_biexp": IAR_LU_biexp,
        "IAR_LU_segmented_2step": IAR_LU_segmented_2step,
        "IAR_LU_segmented_3step": IAR_LU_segmented_3step,
        "IAR_LU_subtracted": IAR_LU_subtracted,
    }

    for name, algorithm in algorithms.items():
        try:
            algo_instance = algorithm()
            maps = OsipiBase.osipi_fit(algo_instance, data_norm, bval)

            f_array, Dstar_array, D_array = maps["f"], maps["D*"], maps["D"]

            # Initialize full-size arrays
            f_map, Dstar_map, D_map = np.zeros(sx * sy * sz), np.zeros(sx * sy * sz), np.zeros(sx * sy * sz)
            f_map[valid_id], Dstar_map[valid_id], D_map[valid_id] = f_array, Dstar_array, D_array

            # Save results to CSV
            suffix = f'_slice{slice_idx}' if slice_idx is not None else ''
            df = pd.DataFrame({"f": f_map, "D*": Dstar_map, "D": D_map})
            df.to_csv(f'{base_dir}/IVIM_{anatomy}_{mask_name}_{name}{suffix}.csv', index=False)
            print(f'Saved results for {name} ({anatomy} - {mask_name})')
        except Exception as e:
            print(f'Error with {name}: {e}')


if __name__ == "__main__":
    download_data()
    slice_idx = 10  # Set to an integer (e.g., 30) to process only a single 2D slice
    base_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(base_dir, exist_ok=True)
    for anatomy in ["brain", "abdomen"]:
        bval, data = load_data(anatomy)
        masks = load_mask(anatomy)

        for mask_name, mask in masks.items():
            data_norm, valid_id, sx, sy, sz = preprocess_data(data, bval, mask, slice_idx)
            run_algorithms(data_norm, bval, valid_id, sx, sy, sz, anatomy, base_dir, mask_name, slice_idx)

