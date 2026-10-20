# Abdomen IVIM Preprocessing Pipeline

**Author:** Siria Pasini  
**Institution:** Istituto di Ricerche Farmacologiche Mario Negri IRCCS, Milan, Italy

---

## Overview

This module implements a full preprocessing pipeline for abdomen IVIM (Intravoxel Incoherent Motion) diffusion-weighted MRI data. It is designed for abdominal organs (e.g. kidneys, liver) where respiratory motion, susceptibility distortion, and RF inhomogeneity are the main sources of artefacts.

The pipeline is organised as a set of independent step functions plus a master function (`abdomen_preproc`) that chains them in order.

| Step | Function | Tool |
|------|----------|------|
| 1 | Denoising | MRtrix3 `dwidenoise` |
| 2 | Gibbs ringing removal | MRtrix3 `mrdegibbs` |
| 3 | Motion correction | dipy **or** MIRTK (selectable) |
| 4 | Susceptibility distortion correction | FSL `topup` / `applytopup` |
| 5 | N4 bias field correction | SimpleITK |

---

## Requirements

### Python packages

```bash
pip install nibabel numpy SimpleITK
```

For the **dipy** motion correction backend (default):

```bash
pip install dipy
```

> The MIRTK backend does not require additional Python packages beyond `nibabel` and `numpy`.

### External command-line tools

These must be installed and available on your `PATH`:

| Tool | Purpose | Install |
|------|---------|---------|
| **MRtrix3** | `dwidenoise` (step 1), `mrdegibbs` (step 2) | [mrtrix.readthedocs.io](https://mrtrix.readthedocs.io/en/latest/installation/package_install.html) |
| **FSL** | `fslroi`, `fslmerge`, `topup`, `applytopup` (steps 3–4) | [fsl.fmrib.ox.ac.uk](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) |
| **MIRTK** | `mirtk register`, `mirtk transform-image` (step 3, optional backend) | [mirtk.github.io](https://mirtk.github.io/install.html) |

> **MIRTK is only required if you select `moco_backend='mirtk'`.** The default backend (`'dipy'`) does not need MIRTK.

---

## Input files

| Argument | Description |
|----------|-------------|
| `im_file` | 4D DWI NIfTI (`.nii.gz`) — the raw acquisition |
| `bval_file` | b-value file (`.bval`) matching `im_file` |
| `b0rev_file` | Reversed phase-encoding b=0 NIfTI (`.nii.gz`) — required for topup |
| `b0rev_bval_file` | b-value file for the reversed b=0 acquisition |

---

## Usage

```python
from abdomen_preproc import abdomen_preproc

im_file         = '/path/to/abdomen_dwi.nii.gz'
bval_file       = '/path/to/abdomen_dwi.bval'
b0rev_file      = '/path/to/abdomen_b0_rev.nii.gz'
b0rev_bval_file = '/path/to/abdomen_b0_rev.bval'

# Default: dipy motion correction, A→P phase encoding
abdomen_preproc(im_file, bval_file, b0rev_file, b0rev_bval_file,
                moco_backend='dipy',
                pe_direction='j')

# MIRTK rigid registration
abdomen_preproc(im_file, bval_file, b0rev_file, b0rev_bval_file,
                moco_backend='mirtk',
                mirtk_method='rigid',
                pe_direction='j')

# MIRTK affine registration
abdomen_preproc(im_file, bval_file, b0rev_file, b0rev_bval_file,
                moco_backend='mirtk',
                mirtk_method='affine',
                pe_direction='j')
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `moco_backend` | `'dipy'` | Motion correction backend: `'dipy'` or `'mirtk'` |
| `moco_ref_idx` | `0` | Reference volume index for motion correction |
| `mirtk_method` | `'rigid'` | Registration model for MIRTK: `'rigid'` or `'affine'` |
| `pe_direction` | `'j'` | Phase-encoding direction for topup (`'j'`=A→P, `'j-'`=P→A, `'i'`=L→R, `'i-'`=R→L) |
| `total_readout_time` | `0.050` | Total EPI readout time in seconds — **must match your acquisition protocol** |

---

## Output

Each step produces an intermediate NIfTI file with a descriptive suffix appended to the input filename:

```
abdomen_dwi.nii.gz                              ← input
abdomen_dwi-pca.nii.gz                          ← after denoising
abdomen_dwi-pca-gib.nii.gz                      ← after Gibbs removal
abdomen_dwi-pca-gib-moco-dipy.nii.gz            ← after motion correction (dipy)
abdomen_dwi-pca-gib-moco-dipy-unwarp.nii.gz     ← after topup
abdomen_dwi-pca-gib-moco-dipy-unwarp-n4.nii.gz  ← final output
```

The final output path is returned by `abdomen_preproc()`.

---

## Additional utilities

The module also provides two standalone functions for post-correction quality control:

- **`detect_signal_outliers(im_file)`** — flags volumes whose signal deviates from a log-linear ADC model (motion-corrupted volumes).
- **`remove_outlier_volumes(im_file, outlier_mask)`** — removes flagged volumes and saves a cleaned NIfTI with matching `.bval`/`.bvec` files.

These are not called automatically by `abdomen_preproc()` and can be applied at any stage.

---

## Notes

- The `total_readout_time` and `pe_direction` parameters must match your scanner protocol exactly. Incorrect values will produce wrong topup field maps.
- N4 bias correction is applied **volume-by-volume** rather than on the full 4D stack, because signal intensity varies strongly across b-values in IVIM acquisitions.
- The MIRTK motion correction uses a **two-stage strategy**: intra-shell registration with NCC (same contrast), followed by iterative template registration with NMI. Unstable volumes (detected by SI centroid analysis) receive an extra resolution level.
