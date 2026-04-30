import numpy as np
import os
import subprocess
import tempfile
import nibabel as nib

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.denoise.localpca import mppca
from dipy.align.imaffine import AffineRegistration
from dipy.align.transforms import RigidTransform3D


def create_preproc_config():


def read_preproc_config():


def preproc_brain(ivim_file, config_file):
    """
    template for preprocessing in brain according to the reccomendations paper
    """
    data, bvals, bvecs, affine = load_data(ivim_file)

    preproc_ivim(options_brain)

    return ivim_preproc


def preproc_non_brain():
    """
    template for preprocessing in non-brain according to the reccomendations paper
    """
    load_data()


def load_data(data_path):
    """
    loads files and data from the given path. Expects nifti files, and corresponding bval and bvec files with the same name but different extensions.
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


def preproc_ivim(data, affine, bvals, bvecs, motion_cor=True, distortion_cor=True, denoising=True, signal_void=True, config=None):
    """
    Preprocessing steps to be included. The preproc_brain and preproc_non_brain functions will call this function with the appropriate options for each case. 
    The parameters are set in the config dictionary. The steps are chained in order.
    """
    cfg = config or {}
    if denoising:
        data = denoise(data, **cfg.get("denoising", {}))

    if motion_cor:
        data = motion_correction(data, affine, bvals, bvecs, **cfg.get("motion_correction", {}))

    if distortion_cor:
        data = distortion_correction(data, affine, bvals, bvecs, **cfg.get("distortion_correction", {}))

    if signal_void:
        data = signal_void_exclusion(data, **cfg.get("signal_void_exclusion", {}))

    return data

# ________________________________________________________________________________________________________________
# this section contains denoising, motion correction, and signal void exclusion
# nothing is tested...

def denoise(data, **kwargs):
    """ CURENTLY PLACEHOLDER FUNCTION, NEEDS FIXING"""
    return mppca(data, **kwargs)

def motion_correction(data, affine, bvals, bvecs, **kwargs):
    """ CURENTLY PLACEHOLDER FUNCTION, NEEDS FIXING"""
    # rigid/affine registration
    # use the b0 image as reference for registration
    b0_mask = bvals == 0
    if np.sum(b0_mask) == 0:
        raise Exception("No b0 images found. Motion correction requires at least one b0 image.")
    b0_data = data[..., b0_mask]
    reference = np.mean(b0_data, axis=3)

    # perform registration for each volume
    reg = AffineRegistration()
    transform = RigidTransform3D()
    metric = reg.metric.MutualInformation()
    level_iters = [10000, 1000, 100]
    for i in range(data.shape[3]):
        moving = data[..., i]
        affine_map = reg.optimize(reference, moving, transform, metric, level_iters=level_iters)
        data[..., i] = affine_map.transform(moving)

    return data


def signal_void_exclusion(data, **kwargs):
    """ CURENTLY PLACEHOLDER FUNCTION, NEEDS FIXTING"""
    # identify signal voids and exclude them from further analysis
    # currently basic thresholding.
    threshold = kwargs.get("threshold", 0.1)  # example threshold value
    mask = data < threshold
    data[mask] = 0  # set signal voids to zero

    return data


# ________________________________________________________________________________________________________________
# this section contains distortion correction
# not nipype, but subprocess calls FSl. this is because using nipypr requires a lot of setup, and we only use nipype for distcorr and it is not worth.


def distortion_correction(data, affine, bvals, bvecs, blip_down_path=None, acqparams_path=None, working_dir=None, **kwargs):
    """not tested"""
    # Use topup + applytopup when input data matches? (in which case many more inputs are needed...) 
    # otherwise use registration based approach. Here we will implement a simple registration based approach as a placeholder.
    use_topup = (blip_down_path is not None) #assumes that user wants to run topup if bd is provided.
    if use_topup:
        if acqparams_path is None:
            raise ValueError("Acquisition parameters file is not provided. Compulsory for topup.") 
        
        data_dist_corr = topup_distcorr(data, bvals, blip_down_path=blip_down_path, acqparams_path=acqparams_path, working_dir=working_dir)
    else:
        print("[distortion_correction] No blip-down image provided. Running registration-based distortion correction.")
        data_dist_corr = registration_distcorr(data, affine, bvals, bvecs, **kwargs)

    return data_dist_corr

def load_bd(blip_down_path):
    data_bd, bvals_bd, _, _ = load_data(blip_down_path)
    if not np.all(bvals_bd <= 50):
        print("[distortion_correction] Warning: Blip-down contains volumes with bval > 50. Only b0 <= 50 volumes will be used.")
        b0_mask = bvals_bd <= 50
        data_bd = data_bd[..., b0_mask]
    return data_bd

def get_b0(data, bvals, b0_threshold=50):
    b0_mask = bvals <= b0_threshold
    if not np.any(b0_mask):
        raise ValueError("No b0 volumes found in data. Cannot run topup.")
    b0_indices = np.where(b0_mask)[0] # for the index file (applytopup)
    b0_data = data[..., b0_mask]
    return b0_data, b0_indices

def build_imain(b0_up, b0_down):
    if b0_up.shape[:3] != b0_down.shape[:3]:
        raise ValueError(
            f"Blip-up and blip-down volumes have different spatial dimensions: "
            f"{b0_up.shape[:3]} vs {b0_down.shape[:3]}")
    if b0_data_down.ndim == 3:
        b0_data_down = b0_data_down[..., np.newaxis]

    imain = np.concatenate([b0_up, b0_data_down], axis=-1)
    n_up = b0_up.shape[-1] # for index file
    n_down = b0_down.shape[-1]

    return imain, n_up, n_down

def build_index_file(bvals, nup, index_file_path):
    indices = ["1"] * len(bvals)
    with open(index_file_path, "w") as f:
        f.write(" ".join(indices) + "\n")

def topup_distcorr(data, affine, bvals, blip_down_path, acqparams_path, working_dir):
    """
    Runs FSL topup + applytopup.
    1. Extract b0s from the blip-up series.
    2. Load blip-down b0s.
    3. Concatenate into --imain stack (blip-up b0s first, blip-down b0s second).
    4. Run topup.
    5. Apply field to full series with applytopup.
    """
    tmpdir = working_dir or tempfile.mkdtemp(prefix="topup_")
    os.makedirs(tmpdir, exist_ok=True)

    b0_up, b0_indices = get_b0(data, bvals)
    #print(f"[topup_correction] Extracted {b0_up.shape[-1]} b0 volumes from input data for topup.")
    b0_down = load_bd(blip_down_path)
    #print(f"[topup_correction] Loaded blip-down data with shape {b0_down.shape} for topup.")
    imain_data = n_up, n_down = build_imain(b0_up, b0_down)
    imain_path = os.path.join(tmpdir, "imain.nii.gz")
    nib.save(Nifti1Image(imain_data.astype(np.float32), affine), imain_path)
    #print(f"[distortion_correction] topup imain: {n_up} blip-up b0s, {n_down} blip-down b0s")

    field_out  = os.path.join(tmpdir, "topup_field")
    movpar_out = os.path.join(tmpdir, "topup_movpar.txt")

    topup_cmd = [
        "topup",
        f"--imain={imain_path}",
        f"--datain={acqparams_path}",
        "--config=b02b0.cnf",
        f"--out={field_out}",
        f"--movpar={movpar_out}"
    ]
    print("[distortion_correction] Running topup with command: " + " ".join(topup_cmd))
    subprocess.run(topup_cmd, check=True)

    series_path = os.path.join(tmpdir, "data_series.nii.gz")
    corrected_path = os.path.join(tmpdir, "distortion_corrected.nii.gz")
    index_path = os.path.join(tmpdir, "index.txt")
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), series_path)
    build_index_file(bvals, n_up, index_path)
    
    applytopup_cmd = [
        "applytopup",
        f"--imain={series_path}",
        f"--datain={acqparams_path}",
        f"--inindex={index_path}",
        f"--topup={field_out}",
        f"--out={corrected_path}",
        "--method=jac",
    ]

    print("[distortion_correction] Running applytopup with command: " + " ".join(applytopup_cmd))
    subprocess.run(applytopup_cmd, check=True)

    data_corrected, _ = load_nifti(corrected_path)
    return data_corrected


def registration_distcorr(data, affine, bvals, b0_threshold=50, **kwargs):
    """
    THIS IS EXAMPLE FUNCTION!!!! NEEDS FIXING.
    Fallback distortion correction using registration to the mean b0.
    Less accurate than topup but requires no reverse PE acquisition.

    Each volume is registered to the mean b0 using a symmetric
    diffeomorphic approach (SyN) via dipy.
    """
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CCMetric

    b0_data, _ = _extract_b0s(data, bvals, b0_threshold)
    reference = np.mean(b0_data, axis=-1)

    metric = CCMetric(3)
    sdr = SymmetricDiffeomorphicRegistration(metric)

    data_corrected = np.copy(data)
    print("[distortion] Running registration-based correction ...")
    for i in range(data.shape[-1]):
        moving = data[..., i]
        mapping = sdr.optimize(reference, moving)
        data_corrected[..., i] = mapping.transform(moving)

    return data_corrected


#_______________________________________________________________________________________________________________


