import numpy as np
import os

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.denoise.localpca import mppca
from dipy.align.imaffine import AffineRegistration
from dipy.align.transforms import RigidTransform3D


def create_preproc_config():


def read_preproc_config():


def preproc_brain(ivim_file, config_file):
    """

    """
    data, bvals, bvecs, affine = load_data(ivim_file)

    preproc_ivim(options_brain)

    return ivim_preproc


def preproc_non_brain():
    """

    """
    load_data()


def load_data(data_path):
    """

    """
    pre, ext = os.path.splitext(data_path)
    if ext != ".nii" and ext != ".gz":
        raise Exception("Nifti file expected. Got " + ext)
    data, affine = load_nifti(data_path)

    if ext == '.nii':
        bval_file = os.rename(data_path, pre + '.bval')
        bvec_file = os.rename(data_path, pre + '.bvec')
    else:
        pre, tmp = os.path.splitext(pre)
        bval_file = os.rename(data_path, pre + '.bval')
        bvec_file = os.rename(data_path, pre + '.bvec')

    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)

    return data, bvals, bvecs, affine


def preproc_ivim(data, bmat, motion_cor=True, distortions=True, denoising=True):
    """

    """
    # motion correction -> rigid/affine registration
    # distortions -> topup?
    # denoisingk -> mppca
    # signal void exclusion
