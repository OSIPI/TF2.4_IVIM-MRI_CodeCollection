import numpy as np
import csv
import datetime
import nibabel as nib
import os
import pytest
from src.wrappers.OsipiBase import OsipiBase


def phantom_to_array(phantom, folder):
    nii_fn = f"output_{phantom}.nii.gz"
    data = nib.load(os.path.join(folder, nii_fn)).get_fdata()
    n_voxels = np.prod(data.shape[:-1])
    n_time   = data.shape[-1]
    signal_array = data.reshape((n_voxels, n_time))

    GT_D  = nib.load(os.path.join(folder, f"D_{phantom}.nii.gz")).get_fdata().ravel()
    GT_f  = nib.load(os.path.join(folder, f"f_{phantom}.nii.gz")).get_fdata().ravel()
    GT_Dp = nib.load(os.path.join(folder, f"Dp_{phantom}.nii.gz")).get_fdata().ravel()

    bvals  = np.loadtxt(os.path.join(folder, f"bvals_{phantom}.txt"))
    return signal_array, GT_D, GT_f, GT_Dp, bvals


@pytest.mark.phantom
def test_fit_phantom_data(
        algorithmlist,
        save_file,
        save_duration_file,
        use_prior,
        eng,
        phantom,
        folder):

    """Fit IVIM model to each voxel of a real phantom dataset."""
    ivim_algorithm, requires_matlab, deep_learning = algorithmlist
    if requires_matlab and eng is None:
        pytest.skip("Running without Matlab; if Matlab is available run pytest --withmatlab")
    if deep_learning:
        pytest.skip("Deep-learning IVIM fitting not implemented for phantom test")

    signal_array, GT_D, GT_f, GT_Dp, bvals = phantom_to_array(phantom, folder)
    fit = OsipiBase(algorithm=ivim_algorithm)
    phantom = 'original'
    folder = r'C:\\TF_IVIM_OSIPI\\TF2.4_IVIM-MRI_CodeCollection\\phantoms\\MR_XCAT_qMRI'
    total_time = datetime.timedelta()      # <-- initialize once
    n_fitted   = 0

    for idx, signal in enumerate(signal_array):
        if not np.any(signal):              # skip background
            continue
        start_time = datetime.datetime.now()
        result = fit.osipi_fit(signal, bvals)
        total_time += datetime.datetime.now() - start_time
        n_fitted += 1

        if save_file is not None:
            save_file.writerow([
                ivim_algorithm,
                idx,
                GT_f[idx], GT_Dp[idx], GT_D[idx],      # ground truth
                result["f"], result["Dp"], result["D"] # fitted values
            ])

    if save_duration_file is not None and n_fitted > 0:
        mean_us = (total_time / n_fitted) / datetime.timedelta(microseconds=1)
        save_duration_file.writerow([ivim_algorithm, phantom, mean_us, n_fitted])


def save_results(filename, algorithm, idx, truth, fit):
    with open(filename, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([algorithm, idx, *truth, *fit])


def save_duration(filename, algorithm, duration, count):
    with open(filename, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([algorithm, duration/datetime.timedelta(microseconds=1), count])
