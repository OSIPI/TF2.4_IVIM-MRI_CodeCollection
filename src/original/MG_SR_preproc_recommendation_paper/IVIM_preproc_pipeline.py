import numpy as np
import os
import nibabel as nib
import subprocess
import tempfile
import shutil

from IVIM_preproc_config import IVIMPreprocConfig

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.denoise.localpca import mppca, localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D, AffineTransform3D
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric


"""MUST BE TESTED!"""

DENOISE_METHODS = {
    "mppca": mppca,
    "localpca": localpca,
}

#________________________________________________________________________________________________________________
# MAIN RUNNER

def preproc_ivim(datapath, config=None):
    """ WIP: We could include "body region" here and double-check if it's consistent
    with the provided config file....

    Preprocessing pipeline for IVIM data.
    The parameters are set in the config dictionary. The config can be generated using
    IVIMPreprocConfig.create() and IVIMPreproConfig.save().

    The pipeline supports the pre-processing steps suggested in the recommendations
    paper, including denoising, motion correction via registration, distortion
    correction, and signal void exclusion.
    The steps are chained in order as suggested by the recommendations paper:
    Towards clinical translation of intravoxel incoherent motion MRI: Acquisition
    and analysis consensus recommendations. Sigmund et al., JMRI 2026

    Args:
        datapath:   Path to the IVIM data. Expected to be in nifti format
        config:     Path to the *.yml config file. If not provided, currently a default config
                    file (for brain) will be created and used.

    Returns:
        data:       Pre-processed data (np.array)
        bvals:      b-values (np.array)
        bvecs:      b-vectors (np.array)
        affine:     Affine transformation matrix (np.array)
    """

    data, bvals, bvecs, affine = load_data(datapath)
    cfg = IVIMPreprocConfig.load_or_create(config)
    # NOTE: we want to be able to run separate functions for teting. thereby we add flags 'enabled' to config


    # DENOISING____________________________________
    if cfg.denoise.enabled:
        data, sigma = denoise(data, bvals, bvecs, cfg) #bvals, bvecs for localpca

    # MOTION_CORRECTOIN____________________________
    # WIP: Needs to be adjusted to be compatible with config file
    # motion correction is always enables in the config rn. 
    # get inputs config inputs as kwargs
    if cfg.motion.enabled:
        data = motion(data, affine, bvals, bvecs, **vars(cfg.motion))

    # DISTORTION_CORRECTION________________________
    # WIP: Needs to be adjusted to be compatible with config file
    if cfg.distortion.enabled:
        data = distortion(data, affine, bvals, bvecs,
                                     **vars(cfg.distortion))

    # SIGNAL_VOID_EXCLUSION________________________
    # WIP: Needs to be adjusted to be compatible with config file
    if cfg.signal_void.enabled:
        data = signal_void(data, **vars(cfg.signal_void))

    return data, bvals, bvecs, affine


def load_data(data_path):
    """
    Loads files and data from the given path.
    Expects nifti files, and corresponding bval and bvec files with
    the same name but different extensions (*.bval, *.bvec).
    """
    pre, ext = os.path.splitext(data_path)
    if ext != ".nii" and ext != ".gz":
        raise Exception("Nifti file expected. Got " + ext)
 
    data, affine = load_nifti(data_path)

    if ext == ".gz":
        # data_path was e.g. "scan.nii.gz" -> pre is "scan.nii", strip
        # the remaining ".nii" so bval/bvec paths come out as "scan.bval"
        pre, _ = os.path.splitext(pre)
 
    bval_file = pre + ".bval"
    bvec_file = pre + ".bvec"
 
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
 
    return data, bvals, bvecs, affine


# ________________________________________________________________________________________________________________
# DENOISING

def denoise(data, cfg):
    method = cfg.denoise.method
    method = DENOISE_METHODS[method]
    data, sigma = method(data, patch_radius=cfg.denoise.patch_radius)

    return data, sigma

# ______________________________________________________________________________________________________________
# MOTION CORRECTION

def motion(data, affine, bvals, bvecs, method='rigid', **kwargs):
    # rigid/affine registration
    # use the b0 image as reference for registration
    # paramters are not yet configured in config. not tested.
    b0_mask = bvals == 0
    if np.sum(b0_mask) == 0:
        raise Exception("No b0 images found. Motion correction requires at least one b0 image.")
    b0_data = data[..., b0_mask]
    reference = np.mean(b0_data, axis=-1) #use b0 as reference

    # perform registration for each volume
    metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
    #NOTE NOTE NOTE NOTE level _iters has been decreased for testing was like 10000, 1000 100 before
    level_iters = [50, 10, 5]
    reg = AffineRegistration(metric=metric, level_iters=level_iters)
    if method == "rigid":
        transform = RigidTransform3D()
    elif method == "affine":
        transform = AffineTransform3D()
    else:
        # MotionConfig.validate() already restricts method to
        # {"rigid", "affine"}, so this should be unreachable, but fail
        # loudly rather than silently skipping motion correction if it changes.
        raise ValueError(f"Unsupported motion method: {method}")
 
    data_corrected = np.copy(data)
    for i in range(data.shape[-1]):
        moving = data[..., i]
        affine_map = reg.optimize(
            reference, moving, transform, params0=None,
            static_grid2world=affine, moving_grid2world=affine,
        )
        data_corrected[..., i] = affine_map.transform(moving)
 
    return data_corrected

# ______________________________________________________________________________________________________________
# SIGNAL VOID EXCLUSION

def signal_void(data, **kwargs):
    # identify signal voids and exclude them from further analysis
    # currently basic thresholding.
    threshold = kwargs.get("threshold", 0.1)  # example threshold value
    mask = data < threshold
    data[mask] = 0  # set signal voids to zero

    return data


# ________________________________________________________________________________________________________________
# DISTORTION CORRECTION

# either RPG correction, OR registration against b0.
# subprocess calls FSl. depending on the osipy plan we can include fsl, or not include it.
# topup needs a bunch of checks, thereby we have the check_requrements function...


def check_topup_requirements(bvals, blip_down_path, acqparams_path, working_dir, b0_threshold):
    """check all prerequisited for running topup. if fulfilled run topup, if not, run registration based distortion correction.
    returns a tuple of (is_ready, problems). if True, topup can be run as is.
    """
    problems =[]

    if not blip_down_path:
        problems.append("No blip-down image provided")
    elif not os.path.isfile(blip_down_path):
        problems.append(f"Blip-down image file not found: {blip_down_path}")


    if not acqparams_path:
        problems.append("No acquisition parameters file provided")
    elif not os.path.isfile(acqparams_path):
        problems.append(f"Acquisition parameters file not found: {acqparams_path}")

    # check if there are at least 2 b0 images in the data. b50s allowed, but b0s should be prefferend.. fix later
    if not np.any(np.asarray(bvals) <= b0_threshold):
        problems.append("No b0 images found in the blip-up data, according to bvals.")
        

    # same check for blipdown data:
    if blip_down_path and os.path.isfile(blip_down_path):
        try:
            _, bvals_bd, _, _ = load_data(blip_down_path)
            if not np.any(bvals_bd <= b0_threshold):
                problems.append(
                    f"blip-down image has no b0 volumes (all bval > {b0_threshold}): {blip_down_path}")
        except Exception as exc:
            problems.append(f"could not read blip-down image ({blip_down_path}): {exc}")

    # FSL binaries need to be importable from the shell.
    for tool in ("topup", "applytopup"):
        if shutil.which(tool) is None:
            problems.append(f"FSL tool '{tool}' not found on PATH")

  
    # working_dir, if given, needs to be writable (or creatable). An empty
    # working_dir is fine -- topup_distcorr() falls back to a temp dir.
    if working_dir:
        if os.path.isdir(working_dir):
            if not os.access(working_dir, os.W_OK):
                problems.append(f"working_dir is not writable: {working_dir}")
        else:
            parent = os.path.dirname(os.path.abspath(working_dir)) or "."
            if not os.access(parent, os.W_OK):
                problems.append(f"working_dir does not exist and cannot be created: {working_dir}")
 
    return (len(problems) == 0, problems)


# distortion options function_________________________
def distortion(data, affine, bvals, bvecs, blip_down_path='', acqparams_path='', b0_threshold = 50, working_dir='',
                          **kwargs):

    """Run distortion correction: topup if everything needed for it is
    available, otherwise fall back to registration-based correction.
 
    This function assumes the caller has already decided that distortion
    correction should run at all (see cfg.distortion.enabled in
    preproc_ivim); it is only responsible for picking *which* method to
    use and doing it.
 
    bvecs is currently unused (neither topup nor the registration fallback
    need it), but is kept in the signature for consistency with the other
    pipeline steps and in case a directionally-aware method is added later.
    """

    ready, problems = check_topup_requirements(bvals, blip_down_path, acqparams_path, working_dir, b0_threshold=b0_threshold)
    if ready:
        print("[distortion_correction] All requirements for topup are met. Running topup.")
        data_dist_corr = topup_distcorr(data, affine, bvals,
                                        blip_down_path=blip_down_path,
                                        acqparams_path=acqparams_path,
                                        b0_threshold=b0_threshold,
                                        working_dir=working_dir)

    else:
        print("[distortion_correction] Topup requirements not met. Falling back to registration-based distortion correction.")
        for problem in problems:
            print(f"  - {problem}")
        data_dist_corr = registration_distcorr(data, affine, bvals,
                                               b0_threshold=b0_threshold,
                                               **kwargs)

    return data_dist_corr


# topup_helper_funcions________________________________________
def load_bd(blip_down_path, b0_threshold):
    data_bd, bvals_bd, _, _ = load_data(blip_down_path)
    if not np.all(bvals_bd <= b0_threshold):
        print(f"[distortion_correction] Warning: Blip-down contains volumes with bval > {b0_threshold}. "
              f"Only b0 <= {b0_threshold} volumes will be used for topup.")
        b0_mask = bvals_bd <= b0_threshold
        data_bd = data_bd[..., b0_mask]
    return data_bd


def get_b0(data, bvals, b0_threshold):
    b0_mask = bvals <= b0_threshold
    if not np.any(b0_mask):
        raise ValueError(f"No volumes with bval <= {b0_threshold} found in data. Cannot run topup.")
    b0_indices = np.where(b0_mask)[0]  # for the index file (applytopup)
    b0_data = data[..., b0_mask]
    return b0_data, b0_indices


def build_imain(b0_up, b0_down):
    if b0_up.shape[:3] != b0_down.shape[:3]:
        raise ValueError(
            f"Blip-up and blip-down volumes have different spatial dimensions."
            f"{b0_up.shape[:3]} vs {b0_down.shape[:3]}")
    if b0_down.ndim == 3:
        b0_down = b0_down[..., np.newaxis]

    imain = np.concatenate([b0_up, b0_down], axis=-1)
    n_up = b0_up.shape[-1]  # for index file
    n_down = b0_down.shape[-1]

    return imain, n_up, n_down

# index file is for applytopup? 1 for each bup image in stack, 2 for bdown.
def build_index_file(bvals, nup, index_file_path):
    indices = ["1"] * len(bvals)
    with open(index_file_path, "w") as f:
        f.write(" ".join(indices) + "\n")

# ADD ACQPARAMS BUILDER HERE AS WELL..? would need inputs in config file for PEDs. Can be added later.
#def build_acqparams_file(bvals, nup, acqparams_file_path):
#    # Example implementation - adjust according to your specific needs
#    with open(acqparams_file_path, "w") as f:
#        for i in range(nup):
#            f.write(f"0 0 0 0\n")
#        for i in range(len(bvals) - nup):
#            f.write(f"0 0 0 0\n")

# run_functions________________________________________
def topup_distcorr(data, affine, bvals, blip_down_path, acqparams_path, working_dir, b0_threshold = 50):
    """
    Runs FSL topup + applytopup with the following steps:
    1. Extract b0s from the blip-up series.
    2. Load blip-down b0s.
    3. Concatenate into --imain stack (blip-up b0s first, blip-down b0s second).
    4. Run topup.
    5. Apply field to full series with applytopup.
    """
    # use a working directory for topup.
    tmpdir = working_dir or tempfile.mkdtemp(prefix="topup_")
    os.makedirs(tmpdir, exist_ok=True)

    b0_up, b0_indices = get_b0(data, bvals, b0_threshold=b0_threshold)
    # print(f"[topup_correction] Extracted {b0_up.shape[-1]} b0
    # volumes from input data for topup.")
    b0_down = load_bd(blip_down_path, b0_threshold=b0_threshold)
    # print(f"[topup_correction] Loaded blip-down data with shape
    # {b0_down.shape} for topup.")
    imain_data, n_up, n_down = build_imain(b0_up, b0_down)
    imain_path = os.path.join(tmpdir, "imain.nii.gz")
    nib.save(nib.Nifti1Image(imain_data.astype(np.float32), affine), imain_path)
    print(f"[distortion_correction] topup imain: {n_up} blip-up b0s, "
          f"{n_down} blip-down b0s. b0s decided by b0_threshold={b0_threshold}.")
    

    field_out = os.path.join(tmpdir, "topup_field")
    movpar_out = os.path.join(tmpdir, "topup_movpar.txt")

    topup_cmd = [
        "topup",
        f"--imain={imain_path}",
        f"--datain={acqparams_path}",
        "--config=b02b0.cnf",
        f"--out={field_out}",
        f"--movpar={movpar_out}"
    ]
    print("[distortion_correction] Running topup with command: " +
          " ".join(topup_cmd))
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
 
    print("[distortion_correction] Running applytopup with command: " +
          " ".join(applytopup_cmd))
    subprocess.run(applytopup_cmd, check=True)
 
    data_corrected, _ = load_nifti(corrected_path)
    return data_corrected





def _max_syn_levels(min_spatial_dim, radius=4, max_levels=3):
    """HELPER FOR REGISTRATION DISTCORR
    How many SyN pyramid levels are usable before the coarsest level's
    smallest spatial dimension drops below CCMetric's minimum requirement
    of 2*radius + 1 voxels.
 
    IVIM volumes are routinely thin along one axis (few slices), and each
    pyramid level roughly halves every spatial dimension, so a fixed
    3-level pyramid (dipy's default) can crash on otherwise-normal data --
    this is exactly what happened with a 30-slice volume and radius=4
    (30 -> 15 -> 7, and 7 < 9). This sizes the pyramid to the data instead
    of assuming a level count that only works for large/isotropic volumes.
    """
    min_required = 2 * radius + 1
    levels = 1
    dim = min_spatial_dim
    while dim // 2 >= min_required and levels < max_levels:
        dim //= 2
        levels += 1
    return levels

def registration_distcorr(data, affine, bvals, b0_threshold, **kwargs):
    """
    Each volume is registered to the mean b0 using a symmetric
    diffeomorphic approach (SyN) via dipy.
    """

    b0_data, _ = get_b0(data, bvals, b0_threshold)
    reference = np.mean(b0_data, axis=-1)
 
    radius = 4
    min_spatial_dim = min(reference.shape[:3])
    levels = _max_syn_levels(min_spatial_dim, radius=radius, max_levels=3)
    if levels < 3:
        print(f"[distortion] Volume's smallest spatial dimension is "
              f"{min_spatial_dim}; using {levels} SyN pyramid level(s) "
              f"instead of the default 3 so CCMetric's minimum-size "
              f"requirement (radius={radius}) isn't violated at the "
              f"coarsest level.")
 
    # Iteration counts per level, coarse-to-fine. Not tuned/validated
    # just sized to match the level count computed above instead of crashing.
    level_iters = [50] * (levels - 1) + [25] # NOTE NOTE NOTE NOTE
 
    metric = CCMetric(3, radius=radius)
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters=level_iters)
 
    data_corrected = np.copy(data)
    print("[distortion] Running registration-based correction ...")
    for i in range(data.shape[-1]):
        moving = data[..., i]
        mapping = sdr.optimize(reference, moving)
        data_corrected[..., i] = mapping.transform(moving)
 
    return data_corrected
