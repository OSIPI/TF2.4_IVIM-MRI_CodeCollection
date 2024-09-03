import json
import os
import nibabel as nib
import numpy as np


def read_nifti_file(input_file):
    """
    For reading the 4d nifti image
    """
    nifti_img = nib.load(input_file)
    return nifti_img.get_fdata(), nifti_img.header

def save_nifti_file(data, output_file, affine=None, **kwargs):
    """
    For saving the 3d nifti images of the output of the algorithm
    """
    if affine is None:
        affine = np.eye(4)
    output_img = nib.nifti1.Nifti1Image(data, affine , **kwargs)
    nib.save(output_img, output_file)

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

    bval_data = np.loadtxt(bval_file, dtype=float)
    return bval_data

def read_bvec_file(bvec_file):
    """
    For reading the bvec file
    """
    if not os.path.exists(bvec_file):
        raise FileNotFoundError(f"File '{bvec_file}' not found.")

    bvec_data = np.loadtxt(bvec_file)
    if len(bvec_data.shape) == 2 and bvec_data.shape[1] != 3:
        bvec_data = np.transpose(bvec_data)  # Transpose the array
    return bvec_data
