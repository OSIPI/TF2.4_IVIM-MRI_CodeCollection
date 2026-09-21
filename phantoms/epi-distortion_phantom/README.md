# EPI Distortion Phantom

## Description
A QIBA diffusion phantom acquired on a 3T scanner, intended for testing EPI distortion correction pipelines.

## Contents
| File | Description |
|------|-------------|
| `data/data.nii` | Original DWI acquisition |
| `b0-flipped/data_flipped.nii` | DWI acquisition with flipped phase-encoding direction |
| `data_bvals.bval` | B-values |
| `data_bvecs.bvec` | Gradient directions |

## Usage
The two acquisitions (`data.nii` and `data_flipped.nii`) have opposite phase-encoding directions and can be used as input to EPI distortion correction tools such as FSL's `topup`.

## Notes
- Scanner: 3T
- Phantom: QIBA diffusion phantom
