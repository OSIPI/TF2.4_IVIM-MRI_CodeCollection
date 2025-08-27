#!/usr/bin/env python3
import os
import glob
import json
import nibabel as nib
import shutil
import numpy as np
from tqdm import tqdm
from src.wrappers.OsipiBase import OsipiBase

# --------------------- Helper Functions --------------------- #

def read_nifti_file(input_file):
    nifti_img = nib.load(input_file)
    return nifti_img.get_fdata(), nifti_img.affine, nifti_img.header

def read_bval_file(bval_file):
    return np.genfromtxt(bval_file, dtype=float)

def read_bvec_file(bvec_file):
    bvec_data = np.genfromtxt(bvec_file)
    return np.transpose(bvec_data)

def save_nifti_file(data, affine, output_file):
    output_img = nib.Nifti1Image(data, affine)
    nib.save(output_img, output_file)

def loop_over_first_n_minus_1_dimensions(arr):
    n = arr.ndim
    for idx in np.ndindex(*arr.shape[:n - 1]):
        yield idx, arr[idx].flatten()

def load_config():
    workflow_dir = os.environ["WORKFLOW_DIR"]
    config_path = os.path.join(workflow_dir, "conf", "conf.json")
    with open(config_path, "r") as f:
        return json.load(f)

# --------------------- Main Execution --------------------- #
if __name__ == "__main__":
    # Load config
    config = load_config()
    config = config["workflow_form"]

    # Kaapana environment variables
    WORKFLOW_DIR = os.environ["WORKFLOW_DIR"]
    OPERATOR_IN_DIR = os.environ["OPERATOR_IN_DIR"]
    OPERATOR_OUT_DIR = os.environ["OPERATOR_OUT_DIR"]

    element_input_dir = os.path.join(WORKFLOW_DIR, OPERATOR_IN_DIR)
    element_output_dir = os.path.join(WORKFLOW_DIR, OPERATOR_OUT_DIR)
    os.makedirs(element_output_dir, exist_ok=True)

    # Initialize input_file, bvec_file, bval_file to None
    input_file = None
    bvec_file = None
    bval_file = None

    # Check upload type
    dicom_or_nifti = config.get("upload_type", "nifti").lower()

    if dicom_or_nifti == "nifti":
        # Existing logic for NIfTI uploads
        source_files_kaapana = [f.strip() for f in config.get("source_files").split(",")]
        input_file, bvec_file, bval_file = source_files_kaapana

        input_file = os.path.join(element_input_dir, input_file)
        bvec_file = os.path.join(element_input_dir, bvec_file)
        bval_file = os.path.join(element_input_dir, bval_file)

    else:
        # DICOM case → follow batch structure
        BATCH_DIR = os.path.join(WORKFLOW_DIR, "batch")
        batch_folders = sorted([f for f in glob.glob(os.path.join(BATCH_DIR, "*"))])
        print(f"batch-folders - {batch_folders}")

        if not batch_folders:
            raise FileNotFoundError(f"No batch folders found in {BATCH_DIR}")

        # Pick the first batch folder (usually one per patient/series)
        batch_input_dir = batch_folders[0]
        batch_input_dir = os.path.join(batch_input_dir, "dicom_to_nifti")

        nifti_files = glob.glob(os.path.join(batch_input_dir, "*.nii.gz"))
        bvec_files = glob.glob(os.path.join(batch_input_dir, "*.bvec"))
        bval_files = glob.glob(os.path.join(batch_input_dir, "*.bval"))

        if not nifti_files or not bvec_files or not bval_files:
            raise FileNotFoundError(
                f"Cannot find NIfTI or bvec/bval files in {batch_input_dir}"
            )

        # Take the first matching file
        input_file = nifti_files[0]
        bvec_file = bvec_files[0]
        bval_file = bval_files[0]

        print(f"Using DICOM-converted files from {batch_input_dir}:\n"
              f"  {input_file}\n  {bvec_file}\n  {bval_file}")

    # Optional config values
    affine_override = config.get("affine", None)
    algorithm = config.get("algorithm", "OJ_GU_seg")
    algorithm_args = config.get("algorithm_args", None)

    # Load input data
    data, affine, _ = read_nifti_file(input_file)
    bvecs = read_bvec_file(bvec_file)
    bvals = read_bval_file(bval_file)

    # Override affine if provided
    if affine_override:
        affine = np.array(affine_override).reshape(4, 4)

    # Initialize model
    fit = OsipiBase(algorithm=algorithm)

    # Preallocate output arrays
    shape = data.shape[:data.ndim - 1]
    f_image = np.zeros(shape, dtype=np.float32)
    Dp_image = np.zeros(shape, dtype=np.float32)
    D_image = np.zeros(shape, dtype=np.float32)

    total_iteration = np.prod(shape)

    # Fit IVIM model voxel by voxel
    for idx, view in tqdm(
        loop_over_first_n_minus_1_dimensions(data),
        desc="Fitting IVIM model", dynamic_ncols=True, total=total_iteration
    ):
        fit_result = fit.osipi_fit(view, bvals)
        f_image[idx] = fit_result["f"]
        Dp_image[idx] = fit_result["Dp"]
        D_image[idx] = fit_result["D"]

    # Save outputs
    save_nifti_file(f_image, affine, os.path.join(element_output_dir, "f.nii.gz"))
    save_nifti_file(Dp_image, affine, os.path.join(element_output_dir, "dp.nii.gz"))
    save_nifti_file(D_image, affine, os.path.join(element_output_dir, "d.nii.gz"))

    # Copy all .nii.gz from input directory to WORKFLOW_DIR (Kaapana workaround)
    nii_files = glob.glob(os.path.join(element_input_dir, "*.nii.gz"))
    for nii_file in nii_files:
        shutil.copy(nii_file, WORKFLOW_DIR)
        print(f"Copied {nii_file} to {WORKFLOW_DIR}")

