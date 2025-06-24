import argparse
import json
import os
from pathlib import Path
import nibabel as nib
from src.wrappers.OsipiBase import OsipiBase
import numpy as np
from tqdm import tqdm

def read_nifti_file(input_file):
    """
    For reading the 4d nifti image
    """
    nifti_img = nib.load(input_file)
    return nifti_img.get_fdata(), nifti_img.header

def read_json_file(json_file):
    """
    For reading the json file
    """

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"File '{json_file}' not found.")

    with open(json_file, "r") as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON in file '{json_file}': {e}")

    return json_data

def read_bval_file(bval_file):
    """
    For reading the bval file
    """
    if not os.path.exists(bval_file):
        raise FileNotFoundError(f"File '{bval_file}' not found.")

    bval_data = np.genfromtxt(bval_file, dtype=float)
    return bval_data

def read_bvec_file(bvec_file):
    """
    For reading the bvec file
    """
    if not os.path.exists(bvec_file):
        raise FileNotFoundError(f"File '{bvec_file}' not found.")

    bvec_data = np.genfromtxt(bvec_file)
    bvec_data = np.transpose(bvec_data)  # Transpose the array
    return bvec_data

def save_nifti_file(data, output_file, affine=None, **kwargs):
    """
    For saving the 3d nifti images of the output of the algorithm
    """
    if affine is None:
        affine = np.eye(4)
    else:
        affine = np.array(affine.reshape(4, 4))
    output_img = nib.nifti1.Nifti1Image(data, affine , **kwargs)
    nib.save(output_img, output_file)

def loop_over_first_n_minus_1_dimensions(arr):
    """
    Loops over the first n-1 dimensions of a numpy array.

    Args:
        arr: A numpy array.

    Yields:
        A tuple containing the indices for the current iteration and a flattened view of the remaining dimensions.
    """
    n = arr.ndim
    for idx in np.ndindex(*arr.shape[:n-1]):
        flat_view = arr[idx].flatten()
        yield idx, flat_view



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a 4D NIfTI phantom file along with BIDS JSON, b-vector, and b-value files.")
    parser.add_argument("input_file", type=Path, help="Path to the input 4D NIfTI file")
    parser.add_argument("bvec_file", type=Path, help="Path to the b-vector file.")
    parser.add_argument("bval_file", type=Path, help="Path to the b-value file.")
    parser.add_argument("--affine", type=float, nargs="+", help="Affine matrix for NIfTI image.")
    parser.add_argument("--algorithm", type=str, default="OJ_GU_seg", help="Select the algorithm to use.")
    parser.add_argument("--algorithm_args", nargs=argparse.REMAINDER, help="Additional arguments for the algorithm.")

    args = parser.parse_args()

    try:
        # Read the 4D NIfTI file
        data, _ = read_nifti_file(args.input_file)

        # Read the b-vector, and b-value files
        bvecs = read_bvec_file(args.bvec_file)
        bvals = read_bval_file(args.bval_file)

        # Pass additional arguments to the algorithm

        fit = OsipiBase(algorithm=args.algorithm)
        f_image = []
        Dp_image = []
        D_image = []

        # This is necessary for the tqdm to display progress bar.
        n = data.ndim
        total_iteration = np.prod(data.shape[:n-1])
        for idx, view in tqdm(loop_over_first_n_minus_1_dimensions(data), desc=f"{args.algorithm} is fitting", dynamic_ncols=True, total=total_iteration):
            fit_result = fit.osipi_fit(view, bvals)
            f_image.append(fit_result["f"])
            Dp_image.append(fit_result["Dp"])
            D_image.append(fit_result["D"])

        # Convert lists to NumPy arrays
        f_image = np.array(f_image)
        Dp_image = np.array(Dp_image)
        D_image = np.array(D_image)

        # Reshape arrays if needed
        f_image = f_image.reshape(data.shape[:data.ndim-1])
        Dp_image = Dp_image.reshape(data.shape[:data.ndim-1])
        D_image = D_image.reshape(data.shape[:data.ndim-1])

        save_nifti_file(f_image, "f.nii.gz", args.affine)
        save_nifti_file(Dp_image, "dp.nii.gz", args.affine)
        save_nifti_file(D_image, "d.nii.gz", args.affine)

    except Exception as e:
        print(f"Error: {e}")

