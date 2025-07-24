import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from src.wrappers.OsipiBase import OsipiBase
import glob

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
    for idx in np.ndindex(*arr.shape[:n-1]):
        yield idx, arr[idx].flatten()

if __name__ == "__main__":
    # Read Kaapana environment variables
    WORKFLOW_DIR = os.environ["WORKFLOW_DIR"]
    BATCH_NAME = os.environ["BATCH_NAME"]
    OPERATOR_IN_DIR = os.environ["OPERATOR_IN_DIR"]
    OPERATOR_OUT_DIR = os.environ["OPERATOR_OUT_DIR"]

    # Find all series (batch folders)
    batch_folders = sorted(
        glob.glob(os.path.join("/", WORKFLOW_DIR, BATCH_NAME, "*"))
    )

    # For each series
    for batch_element_dir in batch_folders:
        element_input_dir = os.path.join(batch_element_dir, OPERATOR_IN_DIR)
        element_output_dir = os.path.join(batch_element_dir, OPERATOR_OUT_DIR)

        os.makedirs(element_output_dir, exist_ok=True)

        # Input files
        input_file = os.path.join(element_input_dir, "brain.nii.gz")
        bvec_file = os.path.join(element_input_dir, "brain.bvec")
        bval_file = os.path.join(element_input_dir, "brain.bval")

        # Read inputs
        data, affine, _ = read_nifti_file(input_file)
        bvecs = read_bvec_file(bvec_file)
        bvals = read_bval_file(bval_file)

        # Run IVIM fitting
        fit = OsipiBase(algorithm="OJ_GU_seg")
        f_image, Dp_image, D_image = [], [], []

        total_iteration = np.prod(data.shape[:data.ndim-1])

        for idx, view in tqdm(
            loop_over_first_n_minus_1_dimensions(data),
            desc="Fitting IVIM model", dynamic_ncols=True, total=total_iteration
        ):
            fit_result = fit.osipi_fit(view, bvals)
            f_image.append(fit_result["f"])
            Dp_image.append(fit_result["Dp"])
            D_image.append(fit_result["D"])

        # Convert to numpy and reshape
        shape = data.shape[:data.ndim-1]
        f_image = np.array(f_image).reshape(shape)
        Dp_image = np.array(Dp_image).reshape(shape)
        D_image = np.array(D_image).reshape(shape)

        # Save outputs into OPERATOR_OUT_DIR
        save_nifti_file(f_image, affine, os.path.join(element_output_dir, "f.nii.gz"))
        save_nifti_file(Dp_image, affine, os.path.join(element_output_dir, "dp.nii.gz"))
        save_nifti_file(D_image, affine, os.path.join(element_output_dir, "d.nii.gz"))

