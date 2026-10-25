
#%%
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_fieldmap(shape, max_hz=100.0):
        """
        Generate a synthetic off-resonance field in Hz.

        Assumes array axes:
            0: left-right
            1: anterior-posterior
            2: inferior-superior

        This is a mathematical test field, not a
        subject-specific physical fieldmap.
        """

        x, y, z = np.meshgrid(
            np.linspace(-1, 1, shape[0]),
            np.linspace(-1, 1, shape[1]),
            np.linspace(-1, 1, shape[2]),
            indexing="ij"
        )

        # Localized susceptibility-like perturbations
        frontal = np.exp(
            -((x / 0.45)**2
            + ((y - 0.65) / 0.5)**2
            + ((z + 0.15) / 0.35)**2)
        )

        temporal_left = np.exp(
            -(((x - 0.7) / 0.25)**2
            + (y / 0.35)**2
            + ((z + 0.3) / 0.3)**2)
        )

        temporal_right = np.exp(
            -(((x + 0.7) / 0.25)**2
            + (y / 0.35)**2
            + ((z + 0.3) / 0.3)**2)
        )

        field = (
            1.0 * frontal
            + 1.6 * temporal_left
            + 1.6 * temporal_right
        )

        field = gaussian_filter(field, sigma=2)

        field /= max(np.max(np.abs(field)), 1e-12)

        return field * max_hz
from scipy.ndimage import map_coordinates

def apply_susceptibility_distortion(
    image,
    fieldmap_hz,
    total_readout_time,
    pe_axis=1,
    pe_sign=1,
    iterations=15,
    jacobian_modulation=True
):
    """
    Apply a synthetic EPI susceptibility distortion.

    image: 3D undistorted image
    fieldmap_hz: 3D off-resonance field in Hz
    total_readout_time: seconds
    pe_axis: phase-encoding array axis (0, 1, or 2)
    pe_sign: +1 or -1
    jacobian_modulation: simulate geometric pile-up
                         and stretching-related signal changes

    Returns:
        distorted image
    """

    image = np.asarray(image, dtype=np.float64)

    displacement = (
        pe_sign * fieldmap_hz * total_readout_time
    )

    grid = np.meshgrid(
        *[np.arange(n, dtype=float) for n in image.shape],
        indexing="ij"
    )

    # Solve inverse coordinate mapping:
    # y_distorted = y_source + displacement(y_source)
    source = [g.copy() for g in grid]

    for _ in range(iterations):
        sampled_displacement = map_coordinates(
            displacement,
            source,
            order=1,
            mode="nearest"
        )

        source[pe_axis] = (
            grid[pe_axis] - sampled_displacement
        )

    distorted = map_coordinates(
        image,
        source,
        order=3,
        mode="constant",
        cval=0.0
    )

    if jacobian_modulation:
        # Forward Jacobian along phase encoding
        gradient = np.gradient(
            displacement, axis=pe_axis
        )

        jacobian = 1.0 + gradient

        if np.any(jacobian <= 0):
            raise ValueError(
                "Non-invertible displacement field. "
                "Reduce field strength or readout time."
            )

        sampled_jacobian = map_coordinates(
            jacobian,
            source,
            order=1,
            mode="nearest"
        )

        distorted /= sampled_jacobian

    return distorted

import nibabel as nib
import shutil
base_name = 'phantoms/brain/data/diffusive_snr50'

im = nib.load(base_name + '.nii.gz').get_fdata()
fieldmap=generate_fieldmap(im.shape[:-1],max_hz=100.0)
distorted_im = np.zeros_like(im)
for i in range(im.shape[-1]):
    distorted_im[...,i] = apply_susceptibility_distortion(image=im[...,i],
                                                          fieldmap_hz=fieldmap,
                                                          total_readout_time=0.1,
                                                          pe_axis=1,
                                                          pe_sign=1,
                                                          jacobian_modulation=True)
b0rev = apply_susceptibility_distortion(image=im[...,0],
                                        fieldmap_hz=fieldmap,
                                        total_readout_time=0.1,
                                        pe_axis=1,
                                        pe_sign=-1,
                                        jacobian_modulation=True)
nib.save(nib.Nifti1Image(distorted_im, nib.load(base_name + '.nii.gz').affine), base_name + '_distorted.nii.gz')
shutil.copyfile(base_name + '.bval', base_name + '_distorted.bval')
shutil.copyfile(base_name + '.bvec', base_name + '_distorted.bvec')

slc_idx = im.shape[2]//2-18
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 3))
plt.subplot(1,5,1)
plt.imshow(np.rot90(fieldmap[:,:,slc_idx]))
plt.axis('off')
plt.title('Fieldmap (Hz)')
plt.subplot(1,5,2)
plt.imshow(np.rot90(im[:,:,slc_idx,0]),cmap='gray')
plt.axis('off')
plt.title('Original Image')
plt.subplot(1,5,3)
plt.imshow(np.rot90(distorted_im[:,:,slc_idx,0]),cmap='gray')
plt.axis('off')
plt.title('Distorted Image')
plt.subplot(1,5,4)
plt.imshow(np.rot90(b0rev[:,:,slc_idx]),cmap='gray')
plt.axis('off')
plt.title('B0 Reverse Image')
plt.subplot(1,5,5)
plt.imshow(np.rot90(distorted_im[:,:,slc_idx]-b0rev[:,:,slc_idx]),cmap='RdBu_r',vmin=-0.5,vmax=0.5)
plt.colorbar()
plt.title('Difference Image')
plt.axis('off')

#%% Try applying distortion correction using TF_recommendation_paper pipeline
import nibabel as nib
from src.original.preprocessing.TF_recommendation_paper.IVIM_preproc_pipeline import preproc_ivim
from src.original.preprocessing.TF_recommendation_paper.IVIM_preproc_config import (
    IVIMPreprocConfig,
    DenoiseConfig,
    MotionConfig,
    DistortionConfig,
    SignalVoidConfig,
)
DATA_PATH = "phantoms/brain/data/diffusive_snr50_distorted.nii.gz"              # <-- ADJUST
OUTPUT_PATH = "phantoms/brain/data/diffusive_snr50_distortioncorr.nii.gz"          # <-- ADJUST
CONFIG_PATH = "phantoms/brain/data/diffusive_snr50_config.yml"             # <-- ADJUST

# make config for testing
# enabled = True/False decides if it runs at all. only writing true-> default params.
cfg = IVIMPreprocConfig.create(
    region="brain",
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




