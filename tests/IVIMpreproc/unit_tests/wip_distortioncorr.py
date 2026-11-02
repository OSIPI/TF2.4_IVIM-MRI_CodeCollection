# ----------------------------------------------------------------
# perform denoising using TF_recommendation_paper pipeline
# ----------------------------------------------------------------
import nibabel as nib
import shutil
import numpy as np
 
from src.original.preprocessing.TF_recommendation_paper.IVIM_preproc_pipeline import preproc_ivim
from src.original.preprocessing.TF_recommendation_paper.IVIM_preproc_config import (
    IVIMPreprocConfig,
    DenoiseConfig,
    MotionConfig,
    DistortionConfig,
    SignalVoidConfig,
)

DATA_PATH = "phantoms/brain/data/diffusive_snr50_distorted_motion.nii.gz"              # <-- ADJUST
OUTPUT_PATH = "phantoms/brain/data/diffusive_snr50_distorted_motion_denoised.nii.gz"          # <-- ADJUST
CONFIG_PATH = "phantoms/brain/data/diffusive_snr50_config.yml"             # <-- ADJUST

# make config for testing
# enabled = True/False decides if it runs at all. only writing true-> default params.
cfg = IVIMPreprocConfig.create(
    region="brain",
    denoise=DenoiseConfig(enabled=True),
    motion=MotionConfig(enabled=False), # level_iters in pipeline set to [50,10,5] for speed! NOTE NOTE
    distortion=DistortionConfig(enabled=False), # diffmorphic correction lowk looks weird, has to be looked at.
    signal_void=SignalVoidConfig(enabled=False),
)
cfg.save(CONFIG_PATH)
print(f"Saved config to {CONFIG_PATH}:")
print(cfg.to_dict())

# run with config
data, bvals, bvecs, affine = preproc_ivim(DATA_PATH, config=CONFIG_PATH)
print("Output data shape:", data.shape, "dtype:", data.dtype)
 
nib.save(nib.Nifti1Image(data.astype("float32"), affine), OUTPUT_PATH)
print(f"Saved processed volume to {OUTPUT_PATH}")

# ----------------------------------------------------------------
# perform motion correction using TF_recommendation_paper pipeline
# ----------------------------------------------------------------

shutil.copy(DATA_PATH.replace('.nii.gz','')+'.bval', OUTPUT_PATH.replace('.nii.gz','')+'.bval')
shutil.copy(DATA_PATH.replace('.nii.gz','')+'.bvec', OUTPUT_PATH.replace('.nii.gz','')+'.bvec')

DATA_PATH = "phantoms/brain/data/diffusive_snr50_distorted_motion_denoised.nii.gz"              # <-- ADJUST
OUTPUT_PATH = "phantoms/brain/data/diffusive_snr50_distorted_motion_denoised_motioncorr.nii.gz"          # <-- ADJUST
CONFIG_PATH = "phantoms/brain/data/diffusive_snr50_config.yml"             # <-- ADJUST

# make config for testing
# enabled = True/False decides if it runs at all. only writing true-> default params.
cfg = IVIMPreprocConfig.create(
    region="brain",
    denoise=DenoiseConfig(enabled=False),
    motion=MotionConfig(enabled=True), # level_iters in pipeline set to [50,10,5] for speed! NOTE NOTE
    distortion=DistortionConfig(enabled=False), # diffmorphic correction lowk looks weird, has to be looked at.
    signal_void=SignalVoidConfig(enabled=False),
)
cfg.save(CONFIG_PATH)
print(f"Saved config to {CONFIG_PATH}:")
print(cfg.to_dict())

# run with config
data, bvals, bvecs, affine = preproc_ivim(DATA_PATH, config=CONFIG_PATH)
print("Output data shape:", data.shape, "dtype:", data.dtype)
 
nib.save(nib.Nifti1Image(data.astype("float32"), affine), OUTPUT_PATH)
print(f"Saved processed volume to {OUTPUT_PATH}")

shutil.copy(DATA_PATH.replace('.nii.gz','')+'.bval', OUTPUT_PATH.replace('.nii.gz','')+'.bval')
shutil.copy(DATA_PATH.replace('.nii.gz','')+'.bvec', OUTPUT_PATH.replace('.nii.gz','')+'.bvec')

# ----------------------------------------------------------------
# perform distortion correction using TF_recommendation_paper pipeline
# ----------------------------------------------------------------
DATA_PATH = "phantoms/brain/data/diffusive_snr50_distorted_motion_denoised_motioncorr.nii.gz"              # <-- ADJUST
OUTPUT_PATH = "phantoms/brain/data/diffusive_snr50_distorted_motion_denoised_motioncorr_distortioncorr.nii.gz"          # <-- ADJUST
CONFIG_PATH = "phantoms/brain/data/diffusive_snr50_config.yml"             # <-- ADJUST

working_dir = "phantoms/brain/data"

# Write acqparams.txt file for topup
acqparams_path = "phantoms/brain/data/diffusive_snr50_acqparams.txt"
b = np.loadtxt(DATA_PATH.replace('.nii.gz','')+'.bval')
with open(acqparams_path,'w') as f:
    for _ in range(np.sum(b<=50)):
        f.write('0 1 0 0.050\n')
    # for _ in range(np.sum(brev==0)):
    f.write('0 -1 0 0.050\n')
blip_down_path = "phantoms/brain/data/diffusive_snr50_distorted_b0rev.nii.gz"

# make config for testing
# enabled = True/False decides if it runs at all. only writing true-> default params.
cfg = IVIMPreprocConfig.create(
    region="brain",
    denoise=DenoiseConfig(enabled=False),
    motion=MotionConfig(enabled=False), # level_iters in pipeline set to [50,10,5] for speed! NOTE NOTE
    distortion=DistortionConfig(enabled=True, blip_down_path=blip_down_path,acqparams_path=acqparams_path,working_dir=working_dir), # diffmorphic correction lowk looks weird, has to be looked at.
    signal_void=SignalVoidConfig(enabled=False),
)
cfg.save(CONFIG_PATH)
print(f"Saved config to {CONFIG_PATH}:")
print(cfg.to_dict())

# run with config
data, bvals, bvecs, affine = preproc_ivim(DATA_PATH, config=CONFIG_PATH)
print("Output data shape:", data.shape, "dtype:", data.dtype)
 
nib.save(nib.Nifti1Image(data.astype("float32"), affine), OUTPUT_PATH)
print(f"Saved processed volume to {OUTPUT_PATH}")

shutil.copy(DATA_PATH.replace('.nii.gz','')+'.bval', OUTPUT_PATH.replace('.nii.gz','')+'.bval')
shutil.copy(DATA_PATH.replace('.nii.gz','')+'.bvec', OUTPUT_PATH.replace('.nii.gz','')+'.bvec')