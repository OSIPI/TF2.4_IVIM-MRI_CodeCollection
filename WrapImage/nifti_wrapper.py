import argparse
import json
import os
import sys
import nibabel as nib
from src.wrappers.OsipiBase import OsipiBase
import numpy as np


def read_nifti_file(input_file):
    """
    For reading the 4d nifti image
    """
    nifti_img = nib.load(input_file)
    return nifti_img.get_fdata(), nifti_img.header

def read_nifti_affine(input_file):
    """
    For reading the affine of a nifti image, so that the outputs can be saved in the same space
    """
    return nib.load(input_file).affine

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
        affine = np.array(affine).reshape(4, 4)
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



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Read a 4D NIfTI phantom file along with BIDS JSON, b-vector, and b-value files.")
    parser.add_argument("input_file", type=str, help="Path to the input 4D NIfTI file.")
    parser.add_argument("bvec_file", type=str, help="Path to the b-vector file.")
    parser.add_argument("bval_file", type=str, help="Path to the b-value file.")
    parser.add_argument("--affine", type=float, nargs=16, help="Affine matrix (16 values, row-major) for the output NIfTI images. Defaults to the affine of the input image.")
    parser.add_argument("--algorithm", type=str, default="OJ_GU_seg", help="Select the algorithm to use.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory in which the parameter maps are saved. It is created if it does not exist.")
    parser.add_argument("--mask", type=str, help="Path to a 3D NIfTI mask. Only voxels with a non-zero mask value are fitted; the others are set to zero in the output.")
    parser.add_argument("--body_part", type=str, help="Body part (e.g. brain, liver) used to select literature based default bounds and initial guesses.")
    parser.add_argument("--njobs", type=int, default=1, help="Number of parallel jobs used for the voxel-wise fit. -1 uses all available CPUs.")
    parser.add_argument("--algorithm_args", nargs=argparse.REMAINDER, help="Additional arguments for the algorithm.")
    return parser.parse_args(argv)


def apply_mask(data, mask):
    """
    Marks the voxels outside of the mask as NaN, which OsipiBase.osipi_fit skips and reports as zeros.

    Args:
        data: 4D numpy array, with the b-values along the last dimension.
        mask: numpy array with the shape of the first n-1 dimensions of data.
    """
    if mask.shape != data.shape[:-1]:
        raise ValueError(f"Mask shape {mask.shape} does not match the image shape {data.shape[:-1]}.")
    data = np.array(data, dtype=float)
    data[mask == 0] = np.nan
    return data


def main(argv=None):
    args = parse_args(argv)

    # Read the 4D NIfTI file
    data, _ = read_nifti_file(args.input_file)
    affine = args.affine if args.affine is not None else read_nifti_affine(args.input_file)

    # Read the b-vector, and b-value files
    bvecs = read_bvec_file(args.bvec_file)
    bvals = read_bval_file(args.bval_file)

    if data.shape[-1] != bvals.size:
        raise ValueError(f"The image has {data.shape[-1]} volumes but {bvals.size} b-values were given.")

    if args.mask is not None:
        mask, _ = read_nifti_file(args.mask)
        data = apply_mask(data, mask)

    fit = OsipiBase(algorithm=args.algorithm, bvalues=bvals, body_part=args.body_part)
    results = fit.osipi_fit(data, njobs=args.njobs)

    os.makedirs(args.output_dir, exist_ok=True)
    for key, image in results.items():
        save_nifti_file(image, os.path.join(args.output_dir, f"{key.lower()}.nii.gz"), affine)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
