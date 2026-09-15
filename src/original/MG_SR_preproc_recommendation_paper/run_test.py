"""
Test runner for the IVIM preprocessing pipeline.
Adjust DATA_PATH and the config below as needed.

"""

import nibabel as nib
 
from IVIM_preproc_pipeline import preproc_ivim
from IVIM_preproc_config import (
    IVIMPreprocConfig,
    DenoiseConfig,
    MotionConfig,
    DistortionConfig,
    SignalVoidConfig,
)
 
DATA_PATH = "path/data.nii.gz"              # <-- ADJUST
OUTPUT_PATH = "path/output.nii.gz"          # <-- ADJUST
CONFIG_PATH = "path/config.yml"             # <-- ADJUST

# make config for testing
# enabled = True/False decides if it runs at all. only writing true-> default params.
cfg = IVIMPreprocConfig.create(
    region="body",
    denoise=DenoiseConfig(enabled=False),
    motion=MotionConfig(enabled=False), # level_iters in pipeline set to [50,10,5] for speed! NOTE NOTE
    distortion=DistortionConfig(enabled=True), # diffmorphic correction lowk looks weird, has to be looked at.
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




