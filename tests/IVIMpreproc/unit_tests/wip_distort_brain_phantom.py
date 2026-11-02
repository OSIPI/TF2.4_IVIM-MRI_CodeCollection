# Susceptibility distortion simulation for a brain phantom

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
nib.save(nib.Nifti1Image(b0rev, nib.load(base_name + '.nii.gz').affine), base_name + '_distorted_b0rev.nii.gz')
np.savetxt(base_name+'_distorted_b0rev.bval',np.array([0]))
np.savetxt(base_name+'_distorted_b0rev.bvec',np.array([[0,0,0]]))


# Motion 
import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation 
import nibabel as nib
import shutil

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
base_name = 'phantoms/brain/data/diffusive_snr50'
data = nib.load(base_name+'_distorted.nii.gz').get_fdata()
affine = nib.load(base_name+'_distorted.nii.gz').affine
moved_data = introduce_motion(data,affine)
nib.save(nib.Nifti1Image(moved_data, affine), base_name+'_distorted_motion.nii.gz')
shutil.copy(base_name+'_distorted.bval', base_name+'_distorted_motion.bval')
shutil.copy(base_name+'_distorted.bvec', base_name+'_distorted_motion.bvec')