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
- `--affine`: Affine matrix for NIfTI image (optional).
- `--algorithm`: Select the algorithm to use (default is "OJ_GU_seg").
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
        tf2.4_ivim-mri_codecollection brain.nii.gz brain.bvec brain.bval
    ```

    Replace `brain.nii.gz`, `brain.bvec`, and `brain.bval` with the actual file names you want to use.

---
