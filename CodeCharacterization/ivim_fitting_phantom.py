import numpy as np
import csv
import datetime
import nibabel as nib
import os
from src.wrappers.OsipiBase import OsipiBase
import json


def phantom_to_array(phantom, folder):
    nii_fn = f"output_{phantom}.nii.gz"
    nii_img = nib.load(os.path.join(folder, nii_fn))
    data = nii_img.get_fdata()
    shape3d = data.shape[:-1]   # spatial shape (x,y,z)
    n_voxels = np.prod(shape3d)
    n_time   = data.shape[-1]
    signal_array = data.reshape((n_voxels, n_time))

    GT_D  = nib.load(os.path.join(folder, f"D_{phantom}.nii.gz")).get_fdata().ravel()
    GT_f  = nib.load(os.path.join(folder, f"f_{phantom}.nii.gz")).get_fdata().ravel()
    GT_Dp = nib.load(os.path.join(folder, f"Dp_{phantom}.nii.gz")).get_fdata().ravel()

    bvals = np.loadtxt(os.path.join(folder, f"bvals_{phantom}.txt"))
    return signal_array, GT_D, GT_f, GT_Dp, bvals, shape3d, nii_img.affine, nii_img.header


def fit_phantom_data(
        algorithm_name,
        requires_matlab,
        deep_learning,
        save_file,
        save_duration_file,
        phantom,
        folder,
        eng=None):

    if requires_matlab and eng is None:
        print(f"Skipping {algorithm_name}: requires Matlab")
        return
    if deep_learning:
        print(f"Skipping {algorithm_name}: deep learning not implemented for phantom test")
        return

    signal_array, GT_D, GT_f, GT_Dp, bvals, shape3d, affine, header = phantom_to_array(phantom, folder)

    # Mask: fit only voxels where at least one GT value is nonzero
    fit_mask = (GT_D != 0) | (GT_f != 0) | (GT_Dp != 0)

    # Create result arrays
    fit_f  = np.zeros(signal_array.shape[0])
    fit_Dp = np.zeros(signal_array.shape[0])
    fit_D  = np.zeros(signal_array.shape[0])

    fit = OsipiBase(algorithm=algorithm_name)
    total_time = datetime.timedelta()
    n_fitted = 0

    for idx, signal in enumerate(signal_array):
        if not fit_mask[idx]:
            continue  # skip background voxels

        start_time = datetime.datetime.now()
        result = fit.osipi_fit(signal, bvals)
        total_time += datetime.datetime.now() - start_time
        n_fitted += 1

        # store results
        fit_f[idx]  = float(result["f"])
        fit_Dp[idx] = float(result["Dp"])
        fit_D[idx]  = float(result["D"])

        # write CSV only for fitted voxels
        if save_file is not None:
            save_file.writerow([
                algorithm_name,
                phantom,
                idx,
                float(GT_f[idx]), float(GT_Dp[idx]), float(GT_D[idx]),
                fit_f[idx], fit_Dp[idx], fit_D[idx]
            ])

    # Save duration info
    if save_duration_file is not None and n_fitted > 0:
        mean_us = (total_time / n_fitted) / datetime.timedelta(microseconds=1)
        save_duration_file.writerow([algorithm_name, phantom, mean_us, n_fitted])

    # Reshape to 3D and save nifti
    nib.save(nib.Nifti1Image(fit_f.reshape(shape3d), affine, header),
             fr"phantom_fitting_results\{algorithm_name}_{phantom}_fit_f.nii.gz")
    nib.save(nib.Nifti1Image(fit_Dp.reshape(shape3d), affine, header),
             fr"phantom_fitting_results\{algorithm_name}_{phantom}_fit_Dp.nii.gz")
    nib.save(nib.Nifti1Image(fit_D.reshape(shape3d), affine, header),
             fr"phantom_fitting_results\{algorithm_name}_{phantom}_fit_D.nii.gz")

    print(f"Saved fitted maps and CSV rows for {algorithm_name} ({n_fitted} voxels fitted).")

if __name__ == "__main__":
    # Load JSON
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "..", "tests", "IVIMmodels", "unit_tests", "algorithms.json")
    json_path = os.path.abspath(json_path)
    with open(json_path, "r") as f:
        config = json.load(f)

    # Build algorithm list
    algorithmlist = []
    for name in config["algorithms"]:
        settings = config.get(name, {})
        requires_matlab = settings.get("requires_matlab", False)
        deep_learning   = settings.get("deep_learning", False)
        algorithmlist.append((name, requires_matlab, deep_learning))

    # Phantom and folder
    phantom = "original"
    folder = os.path.join(base_dir, "..", "phantoms", "MR_XCAT_qMRI")
    folder = os.path.abspath(folder)

    # Open CSV files
    with open("test_output_phantom.csv", "w", newline="") as f1, \
         open("test_duration_phantom.csv", "w", newline="") as f2:

        save_file = csv.writer(f1)
        save_duration_file = csv.writer(f2)

        # Write headers
        save_file.writerow(["algorithm", "phantom", "voxel_idx",
                            "GT_f", "GT_Dp", "GT_D",
                            "fit_f", "fit_Dp", "fit_D"])
        save_duration_file.writerow(["algorithm", "phantom", "mean_us_per_voxel", "n_voxels_fit"])

        # Run all algorithms
        for name, requires_matlab, deep_learning in algorithmlist:
            print(f"Running {name} on phantom {phantom}...")
            fit_phantom_data(
                algorithm_name=name,
                requires_matlab=requires_matlab,
                deep_learning=deep_learning,
                save_file=save_file,
                save_duration_file=save_duration_file,
                phantom=phantom,
                folder=folder,
                eng=None
            )