import nibabel as nib
import numpy as np
import os


def phantom_to_array(phantom, folder):
    nii_fn = f"output_{phantom}.nii.gz"
    nii = nib.load(os.path.join(folder, nii_fn))
    data = nii.get_fdata()
    n_voxels = np.prod(data.shape[:-1])
    n_time = data.shape[-1]
    signal_array = data.reshape((n_voxels, n_time))

    nii_GT_D_fn = f"D_{phantom}.nii.gz"
    nii_GT_f_fn = f"f_{phantom}.nii.gz"
    nii_GT_Dp_fn = f"Dp_{phantom}.nii.gz"
    nii_GT_D = nib.load(os.path.join(folder, nii_GT_D_fn))
    nii_GT_f = nib.load(os.path.join(folder, nii_GT_f_fn))
    nii_GT_Dp = nib.load(os.path.join(folder, nii_GT_Dp_fn))

    GT_D = nii_GT_D.get_fdata()
    GT_f = nii_GT_f.get_fdata()
    GT_Dp = nii_GT_Dp.get_fdata()

    GT_D = GT_D.flatten()
    GT_f = GT_f.flatten()
    GT_Dp = GT_Dp.flatten()

    bvals_fn = f"bvals_{phantom}.txt"
    bvals = np.loadtxt(os.path.join(folder, bvals_fn))


    return signal_array, GT_D, GT_f, GT_Dp, bvals

phantom = 'original' # should be a string of ['original', 'one', 'two', 'three', 'four']
folder = r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\phantoms\MR_XCAT_qMRI'
signal, GT_D, GT_f, GT_Dp, bvals = phantom_to_array(phantom, folder)

