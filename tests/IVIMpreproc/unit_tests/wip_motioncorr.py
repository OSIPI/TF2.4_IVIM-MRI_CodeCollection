import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation 
import nibabel as nib

# generate brain phantom with motion artifacts
def introduce_motion(data, affine, max_translation=1.0,max_rotation=2.0,seed=42):
    rng = np.random.default_rng(seed)
    moved_data = np.zeros_like(data)
    for i in range(data.shape[-1]):
        if i == 0:
            moved_data[..., i] = data[..., i]
            continue
        # Random translation and rotation
        translation = rng.uniform(-max_translation, max_translation, size=3)
        angles = rng.uniform(-max_rotation, max_rotation, size=3)  # in degrees
        R = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()

        # Rigid transformation in world coordinates
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation

        # Rotate around the image center
        center_vox = (np.array(data.shape[:3]) - 1) / 2
        center_world = (affine @ np.r_[center_vox, 1])[:3]

        C = np.eye(4)
        C[:3, 3] = center_world

        transform_world = C @ T @ np.linalg.inv(C)

        # Convert to voxel coordinates
        transform_vox = (np.linalg.inv(affine) @ np.linalg.inv(transform_world) @ affine)

        moved_data[..., i] = affine_transform(
            data[..., i],
            transform_vox[:3, :3],
            offset=transform_vox[:3, 3],
            order=3,
            mode='constant',
            cval=0
        )

    return moved_data
data = nib.load('phantoms/brain/data/diffusive_snr50.nii.gz').get_fdata()
affine = nib.load('phantoms/brain/data/diffusive_snr50.nii.gz').affine
moved_data = introduce_motion(data,affine)
nib.save(nib.Nifti1Image(moved_data, affine), 'phantoms/brain/data/diffusive_snr50_motion.nii.gz')

# perform motion correction using TF_recommendation_paper pipeline
import nibabel as nib
 
from src.original.preprocessing.TF_recommendation_paper.IVIM_preproc_pipeline import preproc_ivim
from src.original.preprocessing.TF_recommendation_paper.IVIM_preproc_config import (
    IVIMPreprocConfig,
    DenoiseConfig,
    MotionConfig,
    DistortionConfig,
    SignalVoidConfig,
)

DATA_PATH = "phantoms/brain/data/diffusive_snr50_motion.nii.gz"              # <-- ADJUST
OUTPUT_PATH = "phantoms/brain/data/diffusive_snr50_denoised.nii.gz"          # <-- ADJUST
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



DATA_PATH = "phantoms/brain/data/diffusive_snr50_denoised.nii.gz"              # <-- ADJUST
OUTPUT_PATH = "phantoms/brain/data/diffusive_snr50_motion_corrected.nii.gz"          # <-- ADJUST
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