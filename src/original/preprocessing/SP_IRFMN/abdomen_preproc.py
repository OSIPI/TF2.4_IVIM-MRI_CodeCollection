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
import os as _os
_os.environ['MallocStackLogging'] = '0'
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


def detect_unstable_early_volumes(data, bvals, n_check=12,
                                   z_threshold=2.0):
    '''
    Detect early volumes acquired before breathing stabilised.

    Strategy: estimate the SI (superior-inferior) centroid of signal for
    each of the first n_check volumes. Volumes whose centroid is shifted
    significantly upward (z_threshold MAD above the median) are flagged
    as unstable — the kidney was higher due to irregular respiration.

    Returns
    -------
    unstable_idx : np.ndarray of int — indices of unstable early volumes
    centroids    : np.ndarray of float — SI centroid per checked volume
    '''
    import numpy as _np

    n_check  = min(n_check, data.shape[3])
    slice_ax = 2   # z-axis = SI direction

    # Mean signal per slice for each early volume
    centroids = _np.zeros(n_check)
    for t in range(n_check):
        vol  = data[..., t].astype(_np.float64)
        prof = vol.mean(axis=(0, 1))   # signal profile along z
        prof = _np.maximum(prof, 0)
        total = prof.sum()
        if total > 0:
            centroids[t] = (_np.arange(len(prof)) * prof).sum() / total
        else:
            centroids[t] = len(prof) / 2.0

    # Volumes with centroid significantly ABOVE (higher index = more superior
    # in scanner coords → lower centroid value = shifted up)
    median_c = _np.median(centroids)
    mad_c    = _np.median(_np.abs(centroids - median_c)) + 1e-6
    # Shifted UP = lower centroid value
    unstable = _np.where(centroids < median_c - z_threshold * mad_c)[0]

    print(f'  Early volume centroid analysis (first {n_check} vols):')
    for t in range(n_check):
        flag = ' ← UNSTABLE (shifted up)' if t in unstable else ''
        print(f'    vol {t:3d} (b={int(bvals[t]):4d}): centroid={centroids[t]:.2f}{flag}')
    print(f'  Detected {len(unstable)} unstable early volume(s): {unstable.tolist()}')

    return unstable, centroids


def motion_correction_mirtk(im_file, ref_idx=0, method='rigid'):
    '''
    Two-stage motion correction using MIRTK register.

    Stage 1 — Intra-shell (NCC, same contrast):
      Detect unstable volumes (centroid outliers per shell).
      Register every volume to its shell median (excluding unstable ones
      from the median computation so the reference is stable).

    Stage 2 — Template registration (NMI):
      Build an overall median template from all intra-shell corrected volumes.
      Register every volume to this template.
      Unstable volumes get 5 resolution levels instead of 4.
    '''
    import nibabel as _nib
    import numpy  as _np

    print(f'Step 3 [mirtk/3-stage-{method}]: Motion correction...')

    img    = _nib.load(im_file)
    data   = img.get_fdata(dtype=_np.float32)
    affine = img.affine
    n_vols = data.shape[3]

    bval_file = im_file.replace('.nii.gz', '.bval')
    bvals = _np.genfromtxt(bval_file).flatten() if os.path.isfile(bval_file)             else _np.zeros(n_vols)

    # Sort by b-value
    sort_idx = _np.argsort(bvals, kind='stable')
    if not _np.array_equal(sort_idx, _np.arange(n_vols)):
        print(f'  Reordering volumes by b-value...')
        bvals = bvals[sort_idx]
        data  = data[..., sort_idx]
        _np.savetxt(bval_file, bvals.reshape(1,-1), fmt='%d', delimiter=' ')
        bvec_file = im_file.replace('.nii.gz', '.bvec')
        if os.path.isfile(bvec_file):
            bvecs = _np.genfromtxt(bvec_file)
            if bvecs.ndim == 2 and bvecs.shape[1] == n_vols:
                _np.savetxt(bvec_file, bvecs[:, sort_idx], fmt='%.6f', delimiter=' ')

    unique_bvals = _np.unique(bvals)
    print(f'  Shells before reordering: {bvals.astype(int).tolist()}')
    print(f'  Shells: {unique_bvals.astype(int).tolist()}')

    corrected = data.copy()

    def _save_vol(arr, path):
        _nib.save(_nib.Nifti1Image(arr.astype(_np.float32), affine), path)

    def _mirtk_reg(ref_path, mov_path, dof_path, sim='NMI', levels='4'):
        ret = subprocess.run([
            'mirtk', 'register', ref_path, mov_path,
            '-model',  method.capitalize(),
            '-sim',    sim,
            '-levels', levels,
            '-bg',     '0',
            '-dofout', dof_path,
        ], capture_output=True, text=True)
        return ret.returncode == 0 and os.path.isfile(dof_path)

    def _apply_dof(mov_path, out_path, dof_path, ref_path):
        ret = subprocess.run([
            'mirtk', 'transform-image', mov_path, out_path,
            '-dofin',  dof_path,
            '-target', ref_path,
            '-interp', 'Linear'
        ], capture_output=True, text=True)
        return ret.returncode == 0 and os.path.isfile(out_path)

    def _check_mass(orig, reg, threshold=0.7):
        orig_mass = _np.abs(orig).sum()
        reg_mass  = _np.abs(reg).sum()
        if orig_mass == 0:
            return True, 1.0
        ratio = reg_mass / orig_mass
        return ratio > threshold, ratio

    def _si_centroid(vol):
        prof  = _np.maximum(vol.mean(axis=(0, 1)), 0)
        total = prof.sum()
        return float((_np.arange(len(prof)) * prof).sum() / total)                if total > 0 else vol.shape[2] / 2.0

    # ── Detect unstable volumes from overall centroid distribution ──────────────
    # Overall median is more robust than per-shell median (only 6 vols/shell).
    print(f'\n  Pre-stage: Detecting unstable volumes (overall centroid)...')
    centroids     = _np.array([_si_centroid(data[..., t]) for t in range(n_vols)])
    MAD_THRESHOLD = 2.0

    overall_med   = _np.median(centroids)
    overall_mad   = _np.median(_np.abs(centroids - overall_med)) + 1e-6
    # Initial stable set: within 2 MAD of overall median
    init_stable   = _np.abs(centroids - overall_med) <= 2.0 * overall_mad
    # Unstable = outside the [min, max] range of initially stable centroids
    stable_min    = centroids[init_stable].min() - 0.5
    stable_max    = centroids[init_stable].max() + 0.5
    unstable_mask = (centroids < stable_min) | (centroids > stable_max)
    print(f'  Overall centroid: median={overall_med:.2f}  '
          f'stable range=[{stable_min:.2f}, {stable_max:.2f}]')
    for t in range(n_vols):
        flag = ' ← UNSTABLE' if unstable_mask[t] else ''
        print(f'    vol {t:3d} (b={int(bvals[t]):4d}): centroid={centroids[t]:.2f}{flag}')

    unstable_idx = _np.where(unstable_mask)[0]
    stable_mask  = ~unstable_mask
    print(f'  Total unstable volumes: {len(unstable_idx)} → {unstable_idx.tolist()}')

    # =========================================================================
    # STAGE 1 — Intra-shell registration (NCC)
    # Shell median is built from stable volumes only.
    # =========================================================================
    print(f'\n  Stage 1: Intra-shell registration (NCC)...')

    for bv in unique_bvals:
        idx         = _np.where(bvals == bv)[0]
        stable_idx  = idx[stable_mask[idx]]
        use_idx     = stable_idx if len(stable_idx) >= 1 else idx

        # Shell reference: median of stable volumes
        shell_med = _np.median(data[..., use_idx], axis=-1)
        sref_path = im_file.replace('.nii.gz', f'-s1ref{int(bv)}.nii.gz')
        _save_vol(shell_med, sref_path)

        if len(idx) <= 1:
            os.remove(sref_path)
            continue

        n_levels = '5' if bv >= 200 else '4'
        print(f'    b={int(bv):4d}: {len(idx)} volumes ({len(use_idx)} stable for ref, levels={n_levels})')

        for vi in idx:
            mov = im_file.replace('.nii.gz', f'-s1mov{vi}.nii.gz')
            out = im_file.replace('.nii.gz', f'-s1out{vi}.nii.gz')
            dof = im_file.replace('.nii.gz', f'-s1dof{vi}.dof')
            _save_vol(data[..., vi], mov)
            ok = _mirtk_reg(sref_path, mov, dof, sim='NCC', levels=n_levels)
            if ok:
                ok_t = _apply_dof(mov, out, dof, sref_path)
                if ok_t:
                    reg_arr = _nib.load(out).get_fdata(dtype=_np.float32)
                    ok_mass, ratio = _check_mass(data[..., vi], reg_arr)
                    if ok_mass:
                        corrected[..., vi] = reg_arr
                    else:
                        print(f'      vol {vi}: mass={ratio:.2f} degraded, keeping original')
            for p in [mov, out, dof]:
                if os.path.isfile(p): os.remove(p)
        os.remove(sref_path)

    # =========================================================================
    # STAGE 2 — Iterative template registration (NMI)
    # Register all volumes to overall median template, then iterate:
    # rebuild template → re-detect unstable → re-register still-unstable.
    # =========================================================================
    print(f'\n  Stage 2: Iterative template registration (NMI)...')

    MAX_ITERATIONS      = 3
    CENTROID_STABLE_THR = 2.0

    def _detect_unstable_centroids(data_arr):
        c      = _np.array([_si_centroid(data_arr[..., t])
                             for t in range(data_arr.shape[3])])
        med    = _np.median(c)
        mad    = _np.median(_np.abs(c - med)) + 1e-6
        init_s = _np.abs(c - med) <= CENTROID_STABLE_THR * mad
        if init_s.sum() == 0:
            init_s = _np.ones(len(c), dtype=bool)
        s_min  = c[init_s].min() - 0.5
        s_max  = c[init_s].max() + 0.5
        return (c < s_min) | (c > s_max), c

    def _reg_volumes(vols, corrected_arr, tmpl_p, unstable_cur):
        for t in vols:
            mov = im_file.replace('.nii.gz', f'-s2mov{t}.nii.gz')
            out = im_file.replace('.nii.gz', f'-s2out{t}.nii.gz')
            dof = im_file.replace('.nii.gz', f'-s2dof{t}.dof')
            _save_vol(corrected_arr[..., t], mov)
            n_lev = '5' if (unstable_cur[t] or bvals[t] >= 300) else '4'
            print(f'  Registering volume {t} to reference {ref_idx}...')
            ok = _mirtk_reg(tmpl_p, mov, dof, sim='NMI', levels=n_lev)
            if ok:
                ok_t = _apply_dof(mov, out, dof, tmpl_p)
                if ok_t and os.path.isfile(out):
                    reg_arr = _nib.load(out).get_fdata(dtype=_np.float32)
                    ok_mass, ratio = _check_mass(corrected_arr[..., t], reg_arr)
                    if ok_mass:
                        corrected_arr[..., t] = reg_arr
                    else:
                        print(f'    WARNING: vol {t} degraded (mass={ratio:.2f}), keeping previous')
            for p in [mov, out, dof]:
                if os.path.isfile(p): os.remove(p)

    # Initial template from intra-shell corrected data
    tmpl_arr  = _np.median(corrected, axis=-1).astype(_np.float32)
    tmpl_path = im_file.replace('.nii.gz', '-s2tmpl.nii.gz')
    _save_vol(tmpl_arr, tmpl_path)

    # Pass 1: stable first, unstable last
    vol_order = list(_np.where(stable_mask)[0]) + list(unstable_idx)
    print(f'  Pass 1/{MAX_ITERATIONS}: {n_vols} volumes '
          f'({len(unstable_idx)} unstable with 5 levels)...')
    _reg_volumes(vol_order, corrected, tmpl_path, unstable_mask)

    # Iterative passes: re-register volumes still off centroid
    for iteration in range(2, MAX_ITERATIONS + 1):
        tmpl_arr = _np.median(corrected, axis=-1).astype(_np.float32)
        _save_vol(tmpl_arr, tmpl_path)

        still_unstable, new_centroids = _detect_unstable_centroids(corrected)
        still_idx = _np.where(still_unstable)[0]
        med_c = _np.median(new_centroids)
        mad_c = _np.median(_np.abs(new_centroids - med_c)) + 1e-6
        print(f'  Pass {iteration}/{MAX_ITERATIONS}: centroid median={med_c:.2f} '
              f'MAD={mad_c:.3f} — {len(still_idx)} still unstable: '
              f'{still_idx.tolist()}')

        if len(still_idx) == 0:
            print(f'  ✔ All volumes stable — stopping early.')
            break

        _reg_volumes(still_idx, corrected, tmpl_path, still_unstable)

    if os.path.isfile(tmpl_path): os.remove(tmpl_path)

    # Final report
    _, final_c = _detect_unstable_centroids(corrected)
    med_f = _np.median(final_c)
    mad_f = _np.median(_np.abs(final_c - med_f)) + 1e-6
    final_bad = _np.where(_np.abs(final_c - med_f) > CENTROID_STABLE_THR * mad_f)[0]
    print(f'  Final unstable after registration: '
          f'{final_bad.tolist() if len(final_bad) else "none ✔"}')

    # ── Save output ───────────────────────────────────────────────────────────
    out_file = im_file.replace('.nii.gz', '-moco-mirtk.nii.gz')
    _nib.save(_nib.Nifti1Image(corrected, affine), out_file)
    for ext in ['.bval', '.bvec']:
        src = im_file.replace('.nii.gz', ext)
        dst = out_file.replace('.nii.gz', ext)
        if os.path.isfile(src):
            subprocess.run(['cp', src, dst])
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

def detect_signal_outliers(im_file, bval_threshold=100, rmse_factor=2.5,
                            min_b0_fraction=0.1):
    '''
    Detect DWI volumes whose signal is inconsistent with the expected
    mono-exponential ADC decay, indicating motion corruption.

    Strategy
    --------
    1. Fit a log-linear model  ln(S) = ln(S0) - ADC*b  per voxel using
       only b >= bval_threshold (above perfusion regime).
    2. Compute per-volume RMSE of the fit residuals across all voxels
       with sufficient signal.
    3. Flag volumes whose RMSE exceeds the median + rmse_factor * MAD
       (median absolute deviation — robust to outliers).
    4. Also flag b=0 volumes whose mean signal deviates >3 MAD from the
       median b=0 signal (catches motion-corrupted b=0s).

    Returns
    -------
    outlier_mask : np.ndarray bool (n_vols,) — True = outlier
    rmse_per_vol : np.ndarray float (n_vols,) — per-volume RMSE
    report       : dict with summary statistics
    '''
    import nibabel as _nib
    import numpy  as _np

    img   = _nib.load(im_file)
    data  = img.get_fdata(dtype=_np.float32)
    n_vols = data.shape[3]

    bval_file = im_file.replace('.nii.gz', '.bval')
    bvals = _np.genfromtxt(bval_file).flatten() if os.path.isfile(bval_file)             else _np.zeros(n_vols)

    # Signal mask: voxels with b=0 signal above 10th percentile
    b0_mean = data[..., bvals == bvals.min()].mean(axis=-1)
    sig_mask = b0_mean > _np.percentile(b0_mean[b0_mean > 0], 10)
    n_vox = sig_mask.sum()

    # ── ADC fit on high-b volumes ─────────────────────────────────────────────
    fit_idx  = _np.where(bvals >= bval_threshold)[0]
    b0_idx   = _np.where(bvals == bvals.min())[0]

    rmse_per_vol = _np.zeros(n_vols)

    if len(fit_idx) >= 2:
        S0_mean = data[..., b0_idx].mean(axis=-1)
        S0_flat = S0_mean[sig_mask].astype(_np.float64)

        b_fit   = bvals[fit_idx].astype(_np.float64)
        # Least-squares ADC per voxel across fit_idx volumes
        flat_fit = data[sig_mask][..., fit_idx].astype(_np.float64)  # (n_vox, n_fit)
        with _np.errstate(divide='ignore', invalid='ignore'):
            ratio = flat_fit / S0_flat[:, _np.newaxis]
            log_r = _np.log(_np.where(ratio > 0, ratio, _np.nan))

        # ADC = -sum(b * log_r) / sum(b^2)
        num  = -_np.nansum(b_fit[_np.newaxis,:] * log_r, axis=1)
        den  =  _np.nansum(b_fit[_np.newaxis,:]**2 * _np.isfinite(log_r), axis=1)
        adc  = _np.where(den > 0, num / den, 0.0)        # (n_vox,)

        # Per-volume RMSE of log-residuals across signal voxels
        for vi in fit_idx:
            S      = data[sig_mask, vi].astype(_np.float64)
            with _np.errstate(divide='ignore', invalid='ignore'):
                log_S  = _np.log(_np.where(S > 0, S / S0_flat, _np.nan))
            log_pred   = -adc * bvals[vi]
            resid      = log_S - log_pred
            resid[~_np.isfinite(resid)] = 0.0
            rmse_per_vol[vi] = _np.sqrt(_np.mean(resid**2))

    # ── b=0 outlier detection ─────────────────────────────────────────────────
    b0_signals = _np.array([data[sig_mask, t].mean() for t in b0_idx])
    b0_median  = _np.median(b0_signals)
    b0_mad     = _np.median(_np.abs(b0_signals - b0_median))
    b0_outlier_mask = _np.zeros(n_vols, dtype=bool)
    for i, t in enumerate(b0_idx):
        if _np.abs(b0_signals[i] - b0_median) > 3 * b0_mad:
            b0_outlier_mask[t] = True

    # ── RMSE-based outlier detection ──────────────────────────────────────────
    fit_rmse   = rmse_per_vol[fit_idx]
    rmse_med   = _np.median(fit_rmse)
    rmse_mad   = _np.median(_np.abs(fit_rmse - rmse_med))
    rmse_threshold = rmse_med + rmse_factor * rmse_mad

    rmse_outlier_mask = _np.zeros(n_vols, dtype=bool)
    for t in fit_idx:
        if rmse_per_vol[t] > rmse_threshold:
            rmse_outlier_mask[t] = True

    outlier_mask = rmse_outlier_mask | b0_outlier_mask

    report = {
        'n_vols'         : n_vols,
        'n_outliers'     : int(outlier_mask.sum()),
        'outlier_idx'    : _np.where(outlier_mask)[0].tolist(),
        'rmse_per_vol'   : rmse_per_vol.tolist(),
        'rmse_threshold' : float(rmse_threshold),
        'rmse_median'    : float(rmse_med),
        'rmse_mad'       : float(rmse_mad),
        'b0_outlier_idx' : _np.where(b0_outlier_mask)[0].tolist(),
    }

    print(f'  Signal outlier detection:')
    print(f'    RMSE median={rmse_med:.4f}  MAD={rmse_mad:.4f}  threshold={rmse_threshold:.4f}')
    print(f'    Outlier volumes ({outlier_mask.sum()}): {_np.where(outlier_mask)[0].tolist()}')
    if b0_outlier_mask.any():
        print(f'    b=0 outliers: {_np.where(b0_outlier_mask)[0].tolist()}')

    return outlier_mask, rmse_per_vol, report


def remove_outlier_volumes(im_file, outlier_mask):
    '''
    Remove flagged outlier volumes from the 4-D NIfTI and matching
    .bval / .bvec files. Saves a cleaned file with suffix -clean.nii.gz.

    Returns the path of the cleaned NIfTI.
    '''
    import nibabel as _nib
    import numpy  as _np

    img    = _nib.load(im_file)
    data   = img.get_fdata(dtype=_np.float32)
    affine = img.affine
    n_vols = data.shape[3]

    keep_idx = _np.where(~outlier_mask)[0]
    print(f'  Removing {outlier_mask.sum()} outlier volumes, keeping {len(keep_idx)}/{n_vols}')

    data_clean = data[..., keep_idx]
    out_file   = im_file.replace('.nii.gz', '-clean.nii.gz')
    _nib.save(_nib.Nifti1Image(data_clean, affine), out_file)

    for ext in ['.bval', '.bvec']:
        src = im_file.replace('.nii.gz', ext)
        if not os.path.isfile(src):
            continue
        arr = _np.genfromtxt(src)
        if arr.ndim == 1:
            arr_clean = arr[keep_idx]
            _np.savetxt(out_file.replace('.nii.gz', ext),
                        arr_clean.reshape(1, -1), fmt='%d', delimiter=' ')
        else:
            arr_clean = arr[:, keep_idx]
            _np.savetxt(out_file.replace('.nii.gz', ext),
                        arr_clean, fmt='%.6f', delimiter=' ')

    print(f'  Cleaned file → {out_file}')
    return out_file


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
