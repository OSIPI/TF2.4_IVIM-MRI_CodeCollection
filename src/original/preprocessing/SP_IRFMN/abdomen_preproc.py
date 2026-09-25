'''
Preprocessing of abdomen IVIM data
Siria Pasini — Istituto di Ricerche Farmacologiche Mario Negri IRCCS, Milan, Italy

Overview
--------
This script implements a full preprocessing pipeline for abdomen IVIM (Intravoxel
Incoherent Motion) diffusion-weighted MRI data. It is organized as a set of
independent step functions, each handling one preprocessing stage, plus a master
function (abdomen_preproc) that chains them in order.

The pipeline runs the following steps:

  Step 1 — denoise()
      PCA-based denoising to suppress thermal noise while preserving signal
      structure. Uses MRtrix3 dwidenoise, which exploits the redundancy across
      diffusion directions via Marchenko-Pastur PCA.
      Tool: MRtrix3 (dwidenoise)

  Step 2 — degibbs()
      Gibbs ringing artifact removal using local subvoxel-shifts. Gibbs ringing
      arises from k-space truncation and can bias IVIM parameter estimates,
      particularly at tissue boundaries.
      Tool: MRtrix3 (mrdegibbs)

  Step 3 — motion_correction_dipy() / motion_correction_mirtk()
      Volume-to-volume motion correction, registering each 3D volume to a
      reference (default: first volume, b=0). Two backends are available:

        - motion_correction_dipy  [backend='dipy']
            Pure-Python affine registration using mutual information metric.
            No external binary required. Suitable for moderate motion.
            Tool: dipy (AffineRegistration, MutualInformationMetric)

        - motion_correction_mirtk  [backend='mirtk']
            Rigid or affine registration using the MIRTK command-line tool.
            More robust for large motion; requires MIRTK to be installed.
            Tool: MIRTK (mirtk register), FSL (fslroi, fslmerge)

      The backend is selected via the moco_backend argument of abdomen_preproc().
      When using MIRTK, the registration model ('rigid' or 'affine') is set via
      mirtk_method.

  Step 4 — topup()
      Susceptibility-induced geometric distortion correction using a reversed
      phase-encoding b=0 acquisition. topup estimates the field map from the
      two opposing phase-encoding directions; applytopup corrects the full DWI
      series. The phase-encoding direction and total readout time must match
      your acquisition protocol.
      Tool: FSL (fslroi, fslmerge, topup, applytopup)

  Step 5 — n4_bias_correction()
      N4 bias field correction to remove smooth, low-frequency intensity
      non-uniformities caused by RF coil inhomogeneities. Applied
      volume-by-volume, since signal intensity varies strongly across b-values
      in IVIM acquisitions and a single 4D correction would be unreliable.
      Tool: SimpleITK (N4BiasFieldCorrectionImageFilter)

Entry point
-----------
abdomen_preproc(im_file, bval_file, b0rev_file, b0rev_bval_file, ...)
    Runs all five steps in sequence. See the function docstring for the full
    list of parameters and the commented usage examples at the bottom of the
    script.

Requirements
------------
- MRtrix3   (dwidenoise, mrdegibbs)
- FSL       (fslroi, fslmerge, topup, applytopup)
- SimpleITK  (pip install SimpleITK)
- nibabel, numpy
- dipy      [required only for motion correction backend: dipy]
- MIRTK     [required only for motion correction backend: mirtk]
'''

import os
import subprocess
import numpy as np
import nibabel as nib
import SimpleITK as sitk



# ---------------------------------------------------------------------------
# Step 1 — Denoising
# ---------------------------------------------------------------------------

def denoise(im_file):
    ''' PCA-based denoising using MRtrix3 dwidenoise '''
    im_file_denoised = im_file.replace('.nii.gz', '-pca.nii.gz')
    noise_level      = im_file.replace('.nii.gz', '-noise_level.nii.gz')
    print('Step 1: Denoising...')
    subprocess.run(['dwidenoise', '-noise', noise_level, im_file, im_file_denoised])
    for ext in ['.bval', '.bvec']:
        subprocess.run(['cp', im_file.replace('.nii.gz', ext),
                              im_file_denoised.replace('.nii.gz', ext)])
    return im_file_denoised


# ---------------------------------------------------------------------------
# Step 2 — Gibbs ringing removal
# ---------------------------------------------------------------------------

def degibbs(im_file):
    ''' Gibbs ringing artifact removal using MRtrix3 mrdegibbs '''
    im_file_degibbs = im_file.replace('.nii.gz', '-gib.nii.gz')
    print('Step 2: Gibbs ringing removal...')
    subprocess.run(['mrdegibbs', im_file, im_file_degibbs])
    for ext in ['.bval', '.bvec']:
        subprocess.run(['cp', im_file.replace('.nii.gz', ext),
                              im_file_degibbs.replace('.nii.gz', ext)])
    return im_file_degibbs


# ---------------------------------------------------------------------------
# Step 3 — Motion correction
# ---------------------------------------------------------------------------

def _extract_volume(in_file, t_index):
    ''' Extracts a single 3D volume from a 4D NIfTI using FSL fslroi '''
    out_file = in_file.replace('.nii.gz', f'-tmp{t_index}.nii.gz')
    subprocess.run(['fslroi', in_file, out_file, str(t_index), '1'])
    return out_file


def motion_correction_dipy(im_file, ref_idx=0):
    '''
    Motion correction using dipy affine registration.
    Each volume is registered to ref_idx using mutual information.
    '''
    from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
    from dipy.align.transforms import AffineTransform3D

    print('Step 3 [dipy]: Motion correction...')
    img4d  = nib.load(im_file)
    data   = img4d.get_fdata()
    affine = img4d.affine
    n_vols = data.shape[3]

    ref_data  = data[..., ref_idx]
    corrected = np.zeros_like(data)
    corrected[..., ref_idx] = ref_data

    metric    = MutualInformationMetric(nbins=32, sampling_proportion=None)
    affreg    = AffineRegistration(metric=metric,
                                   level_iters=[10000, 1000, 100],
                                   sigmas=[3.0, 1.0, 0.0],
                                   factors=[4, 2, 1])
    transform = AffineTransform3D()

    for t in range(n_vols):
        if t == ref_idx:
            continue
        print(f'  Registering volume {t} to reference {ref_idx}...')
        affine_map = affreg.optimize(ref_data, data[..., t], transform, None, affine, affine)
        corrected[..., t] = affine_map.transform(data[..., t])

    out_file = im_file.replace('.nii.gz', '-moco-dipy.nii.gz')
    nib.save(nib.Nifti1Image(corrected, affine, img4d.header), out_file)
    for ext in ['.bval', '.bvec']:
        subprocess.run(['cp', im_file.replace('.nii.gz', ext),
                              out_file.replace('.nii.gz', ext)])
    return out_file


def motion_correction_mirtk(im_file, ref_idx=0, method='rigid'):
    '''
    Motion correction using MIRTK register.
    method: 'rigid' or 'affine'
    '''
    print(f'Step 3 [mirtk/{method}]: Motion correction...')
    n_vols  = nib.load(im_file).shape[3]
    tmp_ref = _extract_volume(im_file, ref_idx)

    registered_vols = []
    for t in range(n_vols):
        if t == ref_idx:
            registered_vols.append(tmp_ref)
            continue
        tmp_moving = _extract_volume(im_file, t)
        tmp_out    = im_file.replace('.nii.gz', f'-mirtk-tmp{t}.nii.gz')
        dof_out    = im_file.replace('.nii.gz', f'-mirtk-dof{t}.dof')
        print(f'  Registering volume {t} to reference {ref_idx}...')
        subprocess.run(['mirtk', 'register', tmp_ref, tmp_moving,
                        '-model', method, '-dofout', dof_out, '-output', tmp_out])
        os.remove(tmp_moving)
        registered_vols.append(tmp_out)

    out_file = im_file.replace('.nii.gz', '-moco-mirtk.nii.gz')
    subprocess.run(['fslmerge', '-t', out_file] + registered_vols)

    # Clean up
    os.remove(tmp_ref)
    for t in range(n_vols):
        if t == ref_idx:
            continue
        for f in [im_file.replace('.nii.gz', f'-mirtk-tmp{t}.nii.gz'),
                  im_file.replace('.nii.gz', f'-mirtk-dof{t}.dof')]:
            if os.path.exists(f):
                os.remove(f)

    for ext in ['.bval', '.bvec']:
        subprocess.run(['cp', im_file.replace('.nii.gz', ext),
                              out_file.replace('.nii.gz', ext)])
    return out_file


# ---------------------------------------------------------------------------
# Step 4 — Susceptibility distortion correction
# ---------------------------------------------------------------------------

def topup(im_file, bval_file, b0rev_file, b0rev_bval_file,
          pe_direction='j', total_readout_time=0.050):
    '''
    Susceptibility distortion correction using FSL topup and applytopup.

    pe_direction: phase-encoding direction of the forward acquisition.
        'j'  = A→P,  'j-' = P→A
        'i'  = L→R,  'i-' = R→L
    total_readout_time: total EPI readout time in seconds — check your protocol.
    '''
    _pe_vectors = {'j':  (0,1,0), 'j-': (0,-1,0),
                   'i':  (1,0,0), 'i-': (-1,0,0),
                   'k':  (0,0,1), 'k-': (0,0,-1)}
    fwd = _pe_vectors[pe_direction]
    rev = tuple(-x for x in fwd)

    print('Step 4: Susceptibility distortion correction (topup)...')
    im_file_unwarp = im_file.replace('.nii.gz', '-unwarp.nii.gz')
    interbase      = im_file.replace('.nii.gz', '-b0-b0rev')

    # Load bvals and identify b=0 indices
    b    = np.atleast_1d(np.loadtxt(bval_file))
    brev = np.atleast_1d(np.loadtxt(b0rev_bval_file))
    b0_indices = np.where(b == 0)[0]

    # Extract b=0 volumes from forward acquisition and merge into one file
    b0_base = im_file.replace('.nii.gz', '-b0')
    b0_vols = []
    for i, idx in enumerate(b0_indices):
        tmp = b0_base + f'_tmp{i}.nii.gz'
        subprocess.run(['fslroi', im_file, tmp, str(idx), '1'])
        b0_vols.append(tmp)
    subprocess.run(['fslmerge', '-t', b0_base + '.nii.gz'] + b0_vols)
    for tmp in b0_vols:
        os.remove(tmp)

    # Concatenate forward b=0 and reversed b=0 into one file for topup
    subprocess.run(['fslmerge', '-t', interbase + '.nii.gz',
                    b0_base + '.nii.gz', b0rev_file])

    acqp_file = interbase + '_acqparams.txt'
    with open(acqp_file, 'w') as f:
        for _ in range(int(np.sum(b == 0))):
            f.write(f'{fwd[0]} {fwd[1]} {fwd[2]} {total_readout_time:.3f}\n')
        for _ in range(int(np.sum(brev == 0))):
            f.write(f'{rev[0]} {rev[1]} {rev[2]} {total_readout_time:.3f}\n')

    os.system(f'topup --imain={interbase}.nii.gz --datain={acqp_file} '
              f'--config=b02b0.cnf --subsamp=1 --out={interbase} --verbose')
    os.system(f'applytopup --imain={im_file} --datain={acqp_file} '
              f'--inindex=1 --topup={interbase} --out={im_file_unwarp} '
              f'--method=jac --verbose')

    subprocess.run(['cp', bval_file, im_file_unwarp.replace('.nii.gz', '.bval')])
    return im_file_unwarp


# ---------------------------------------------------------------------------
# Step 5 — N4 bias field correction
# ---------------------------------------------------------------------------

def n4_bias_correction(im_file, n_fitting_levels=4, n_iterations=50,
                       convergence_threshold=0.001):
    '''
    N4 bias field correction (Tustison et al., 2010) applied volume-by-volume
    using SimpleITK. Volume-wise correction is preferred for IVIM data as
    signal intensity varies strongly across b-values.
    '''
    print('Step 5: N4 bias field correction...')
    img4d     = nib.load(im_file)
    data      = img4d.get_fdata()
    n_vols    = data.shape[3]
    corrected = np.zeros_like(data)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([n_iterations] * n_fitting_levels)
    corrector.SetConvergenceThreshold(convergence_threshold)

    for t in range(n_vols):
        print(f'  Correcting volume {t + 1}/{n_vols}...')
        sitk_img = sitk.GetImageFromArray(data[..., t].astype(np.float32).T)
        sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
        mask     = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
        corrected[..., t] = sitk.GetArrayFromImage(
            corrector.Execute(sitk_img, mask)).T

    out_file = im_file.replace('.nii.gz', '-n4.nii.gz')
    nib.save(nib.Nifti1Image(corrected, img4d.affine, img4d.header), out_file)
    subprocess.run(['cp', im_file.replace('.nii.gz', '.bval'),
                          out_file.replace('.nii.gz', '.bval')])
    return out_file


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def abdomen_preproc(im_file, bval_file, b0rev_file, b0rev_bval_file,
                    moco_backend='dipy', moco_ref_idx=0, mirtk_method='rigid',
                    pe_direction='j', total_readout_time=0.050):
    '''
    Full abdomen IVIM preprocessing pipeline.

    Parameters
    ----------
    im_file             : path to the raw 4D DWI NIfTI (.nii.gz)
    bval_file           : path to the corresponding .bval file
    b0rev_file          : path to the reversed phase-encoding b=0 NIfTI
    b0rev_bval_file     : path to the reversed phase-encoding .bval
    moco_backend        : motion correction backend — 'dipy' (default) or 'mirtk'
    moco_ref_idx        : reference volume index for motion correction (default: 0)
    mirtk_method        : 'rigid' (default) or 'affine' — only used when moco_backend='mirtk'
    pe_direction        : phase-encoding direction for topup — 'j', 'j-', 'i', 'i-'
    total_readout_time  : total EPI readout time in seconds for topup (default: 0.050)
    '''

    print('\n=== Step 1/5: Denoising ===')
    im_denoised = denoise(im_file)

    print('\n=== Step 2/5: Gibbs ringing removal ===')
    im_degibbs = degibbs(im_denoised)

    print(f'\n=== Step 3/5: Motion correction [{moco_backend}] ===')
    if moco_backend == 'dipy':
        im_moco = motion_correction_dipy(im_degibbs, ref_idx=moco_ref_idx)
    elif moco_backend == 'mirtk':
        im_moco = motion_correction_mirtk(im_degibbs, ref_idx=moco_ref_idx, method=mirtk_method)
    else:
        raise ValueError(f"moco_backend must be 'dipy' or 'mirtk', got '{moco_backend}'")

    print('\n=== Step 4/5: Susceptibility distortion correction ===')
    im_unwarp = topup(im_moco, bval_file, b0rev_file, b0rev_bval_file,
                      pe_direction=pe_direction,
                      total_readout_time=total_readout_time)

    print('\n=== Step 5/5: N4 bias field correction ===')
    im_final = n4_bias_correction(im_unwarp)

    print(f'\nPipeline complete. Final output: {im_final}')
    return im_final


#%%
# ---------- Example usage ----------
# im_file         = '/path/to/abdomen_dwi.nii.gz'
# bval_file       = '/path/to/abdomen_dwi.bval'
# b0rev_file      = '/path/to/abdomen_b0_rev.nii.gz'
# b0rev_bval_file = '/path/to/abdomen_b0_rev.bval'

# --- with dipy (default) ---
# abdomen_preproc(im_file, bval_file, b0rev_file, b0rev_bval_file,
#                 moco_backend='dipy', pe_direction='j')

# --- with MIRTK rigid ---
# abdomen_preproc(im_file, bval_file, b0rev_file, b0rev_bval_file,
#                 moco_backend='mirtk', mirtk_method='rigid', pe_direction='j')

# --- with MIRTK affine ---
# abdomen_preproc(im_file, bval_file, b0rev_file, b0rev_bval_file,
#                 moco_backend='mirtk', mirtk_method='affine', pe_direction='j')
