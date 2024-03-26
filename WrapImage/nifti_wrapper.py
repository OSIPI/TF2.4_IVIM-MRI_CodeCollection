import argparse
import json
import os
import nibabel as nib
from src.wrappers.OsipiBase import OsipiBase
from utilities.data_simulation.GenerateData import GenerateData
import numpy as np

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

def save_nifti_file(data, output_file, affine=None, **kwargs):
    """
    For saving the 3d nifti images of the output of the algorithm
    """
    if affine is None:
        affine = np.eye(data.ndim + 1)
    output_img = nib.nifti1.Nifti1Image(data, affine , **kwargs)
    nib.save(output_img, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a 4D NIfTI phantom file along with BIDS JSON, b-vector, and b-value files.")
    parser.add_argument("input_file", type=str, help="Path to the input 4D NIfTI file.")
    parser.add_argument("bids_dir", type=str, help="Path to the BIDS directory.")
    parser.add_argument("--affine", type=float, nargs="+", help="Affine matrix for NIfTI image.")
    parser.add_argument("--algorithm", type=str, choices=["algorithm1", "algorithm2"], default="algorithm1", help="Select the algorithm to use.")
    parser.add_argument("algorithm_args", nargs=argparse.REMAINDER, help="Additional arguments for the algorithm.")

    args = parser.parse_args()

    try:
        # Read the 4D NIfTI file
        data, _ = read_nifti_file(args.input_file)

        # Construct the full paths for the JSON, b-vector, and b-value files
        json_file = os.path.join(args.bids_dir, "dataset_description.json")
        bvec_file = os.path.join(args.bids_dir, "bvecs.json")
        bval_file = os.path.join(args.bids_dir, "bvals.json")

        # Read the JSON, b-vector, and b-value files
        json_data = read_json_file(json_file)
        bvecs = read_json_file(bvec_file)
        bvals = read_json_file(bval_file)

        # Pass additional arguments to the algorithm
        fit = OsipiBase(algorithm=args.algorithm)
        # signal = gd.ivim_signal(D, Dp, f, S0, bvals, SNR, rician_noise)

        # Passing the values to the selectect algorithm and saving it
        [f_fit, Dp_fit, D_fit] = fit.osipi_fit(signal, bvals)
        save_nifti_file(f_fit, "f.nii.gz", args.affine)
        save_nifti_file(Dp_fit, "dp.nii.gz", args.affline)
        save_nifti_file(D_fit, "d.nii.gz", args.affline)

    except Exception as e:
        print(f"Error: {e}")
