import os
import glob
import json
import nibabel as nib
import shutil
import numpy as np
from tqdm import tqdm
from src.wrappers.OsipiBase import OsipiBase

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

if __name__ == "__main__":
    # Load config from UI form (mounted at $WORKFLOW_DIR/conf/conf.json)
    config = load_config()
    config = config["workflow_form"]

    # Read Kaapana environment variables
    WORKFLOW_DIR = os.environ["WORKFLOW_DIR"]
    OPERATOR_IN_DIR = os.environ["OPERATOR_IN_DIR"]
    OPERATOR_OUT_DIR = os.environ["OPERATOR_OUT_DIR"]

    element_input_dir = os.path.join(WORKFLOW_DIR, OPERATOR_IN_DIR)
    element_output_dir = os.path.join(WORKFLOW_DIR, OPERATOR_OUT_DIR)

    os.makedirs(element_output_dir, exist_ok=True)

    # Read filenames from config and clean them
    source_files_kaapana = [f.strip() for f in config.get("source_files").split(",")]
    input_file, bvec_file, bval_file = source_files_kaapana

    # Input files
    input_file = os.path.join(element_input_dir, input_file)
    bvec_file = os.path.join(element_input_dir, bvec_file)
    bval_file = os.path.join(element_input_dir, bval_file)

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

    f_image, Dp_image, D_image = [], [], []
    total_iteration = np.prod(data.shape[:data.ndim - 1])

    for idx, view in tqdm(
        loop_over_first_n_minus_1_dimensions(data),
        desc="Fitting IVIM model", dynamic_ncols=True, total=total_iteration
    ):
        fit_result = fit.osipi_fit(view, bvals)
        f_image.append(fit_result["f"])
        Dp_image.append(fit_result["Dp"])
        D_image.append(fit_result["D"])

    # Reshape and save outputs
    shape = data.shape[:data.ndim - 1]
    f_image = np.array(f_image).reshape(shape)
    Dp_image = np.array(Dp_image).reshape(shape)
    D_image = np.array(D_image).reshape(shape)

    save_nifti_file(f_image, affine, os.path.join(element_output_dir, "f.nii.gz"))
    save_nifti_file(Dp_image, affine, os.path.join(element_output_dir, "dp.nii.gz"))
    save_nifti_file(D_image, affine, os.path.join(element_output_dir, "d.nii.gz"))


    # Temporary solution to pass through the kaapana MinioOperator limitation
    # where it uses the same source_files for both put and get actions

    # Find all .nii.gz files in the input directory
    nii_files = glob.glob(os.path.join(element_input_dir, "*.nii.gz"))

    # Copy each file to WORKFLOW_DIR
    for nii_file in nii_files:
        shutil.copy(nii_file, WORKFLOW_DIR)
        print(f"Copied {nii_file} to {WORKFLOW_DIR}")
