import argparse
import json
import os
import nibabel as nib
import numpy as np

def read_nifti_4d(input_file):
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

def save_nifti_3d(data, output_file):
    """
    For saving the 3d nifti images of the output of the algorithm
    """
    output_img = nib.Nifti1Image(data, np.eye(4))
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
        data = read_nifti_4d(args.input_file)

        # Construct the full paths for the JSON, b-vector, and b-value files
        json_file = os.path.join(args.bids_dir, "dataset_description.json")
        bvec_file = os.path.join(args.bids_dir, "bvecs.json")
        bval_file = os.path.join(args.bids_dir, "bvals.json")

        # Read the JSON, b-vector, and b-value files
        json_data = read_json_file(json_file)
        bvecs = read_json_file(bvec_file)
        bvals = read_json_file(bval_file)

        # Pass additional arguments to the algorithm
        # Passing the values to the selectect algorithm and saving it

    except Exception as e:
        print(f"Error: {e}")
