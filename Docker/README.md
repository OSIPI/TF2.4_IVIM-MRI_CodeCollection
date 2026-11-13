# TF2.4_IVIM-MRI_CodeCollection

This project is designed to run the `nifti_wrapper` script using a Docker container. Below are the steps to build and run the Docker image.

## Prerequisites

- Docker must be installed on your system. 

## Directory Structure

```
~/TF2.4_IVIM-MRI_CodeCollection/
│
├── Docker/
│   └── Dockerfile
│
├── WrapImage/
│   └── nifti_wrapper.py
│
└── requirements.txt
```

## Options

Before running the Docker container, here are the available options for the `Docker image` script:

- `input_file`: Path to the input 4D NIfTI file.
- `bvec_file`: Path to the b-vector file.
- `bval_file`: Path to the b-value file.
- `--affine`: Affine matrix (16 values, row-major) for the output NIfTI images (optional). Defaults to the affine of the input image.
- `--algorithm`: Select the algorithm to use (default is "OJ_GU_seg").
- `--output_dir`: Directory in which the parameter maps (`f.nii.gz`, `dp.nii.gz`, `d.nii.gz`) are saved (default is the current directory). It is created if needed.
- `--mask`: Path to a 3D NIfTI mask. Only voxels with a non-zero value are fitted, all others are zero in the output (optional).
- `--body_part`: Body part (e.g. `brain`, `liver`) used to select default bounds and initial guesses (optional).
- `--njobs`: Number of parallel jobs for the voxel-wise fit (default is 1, -1 uses all CPUs).
- `algorithm_args`: Additional arguments for the algorithm (optional).

## Building the Docker Image

1. Open a terminal and navigate to the project directory:

    ```sh
    cd ~/TF2.4_IVIM-MRI_CodeCollection
    ```

2. Build the Docker image using the `docker build` command:

    ```sh
    sudo docker build -t tf2.4_ivim-mri_codecollection -f Docker/Dockerfile .
    ```

## Running the Docker Container

1. Once the image is built, you can run the Docker container using the `docker run` command. This command runs the Docker image with the specified input files:

    ```sh
    sudo docker run -it --rm --name TF2.4_IVIM-MRI_CodeCollection \
        -v ~/TF2.4_IVIM-MRI_CodeCollection:/usr/src/app \
        -v ~/TF2.4_IVIM-MRI_CodeCollection:/usr/app/output \ 
        tf2.4_ivim-mri_codecollection Downloads/brain.nii.gz Downloads/brain.bvec Downloads/brain.bval \
    ```

    Replace `brain.nii.gz`, `brain.bvec`, and `brain.bval` with the actual file names you want to use.
    Add `--output_dir <directory>` to choose where the parameter maps are written, e.g. a mounted folder.

---
