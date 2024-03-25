import argparse
import json
import os
import nibabel as nib
from src.wrappers.OsipiBase import OsipiBase
from utilities.data_simulation.GenerateData import GenerateData
import numpy as np
import random

def read_nifti_file(input_file):
    """
    For reading the 4d nifti image
    """
    nifti_img = nib.load(input_file)
    return nifti_img.get_fdata()

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

def save_nifti_3d(data, output_file, **kwargs):
    """
    For saving the 3d nifti images of the output of the algorithm
    """
    output_img = nib.nifti1.Nifti1Image(data, np.eye(4), **kwargs)
    nib.save(output_img, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a 4D NIfTI phantom file along with BIDS JSON, b-vector, and b-value files.")
    parser.add_argument("input_file", type=str, help="Path to the input 4D NIfTI file.")
    parser.add_argument("bids_dir", type=str, help="Path to the BIDS directory.")
    parser.add_argument("--algorithm", type=str, choices=["algorithm1", "algorithm2"], default="algorithm1", help="Select the algorithm to use.")
    parser.add_argument("algorithm_args", nargs=argparse.REMAINDER, help="Additional arguments for the algorithm.")

    args = parser.parse_args()

    try:
        # Read the 4D NIfTI file
        data = read_nifti_file(args.input_file)

        # Construct the full paths for the JSON, b-vector, and b-value files
        json_file = os.path.join(args.bids_dir, "dataset_description.json")
        bvec_file = os.path.join(args.bids_dir, "bvecs.json")
        bval_file = os.path.join(args.bids_dir, "bvals.json")

        # Read the JSON, b-vector, and b-value files
        json_data = read_json_file(json_file)
        bvecs = read_json_file(bvec_file)
        bvals = read_json_file(bval_file)

        # Pass additional arguments to the algorithm
        rng = np.random.RandomState(42)
        fit = OsipiBase(algorithm=args.algorithm)
        S0 = 1
        gd = GenerateData(rng=rng)
        D = data["D"]
        f = data["f"]
        Dp = data["Dp"]  
        # signal = gd.ivim_signal(D, Dp, f, S0, bvals, SNR, rician_noise)

        # Passing the values to the selectect algorithm and saving it
        [f_fit, Dp_fit, D_fit] = fit.osipi_fit(signal, bvals)
        save_nifti_3d(f_fit, "f.nii.gz")
        save_nifti_3d(Dp_fit, "dp.nii.gz")
        save_nifti_3d(D_fit, "d.nii.gz")

    except Exception as e:
        print(f"Error: {e}")
