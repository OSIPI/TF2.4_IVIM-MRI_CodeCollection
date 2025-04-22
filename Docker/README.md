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
│       └── dicom2nifti/
│           └── Dockerfile
│
├── WrapImage/
│   └── nifti_wrapper.py
│   └── dicom2niix_wrapper.py
│
└── requirements.txt
```

## Options

Before running the Docker container, here are the available options for the `Docker image` script:

- `input_file`: Path to the input 4D NIfTI file or dicom files 4D images.
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
    OR (If you need to convert your data from DICOM TO NIfTI images)
    ```sh
    sudo docker build -t tf2.4_ivim-mri_codecollection -f Docker/dicom2nifti/Dockerfile .
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

## Running the Docker container for reading in DICOM Images

- You can run the dicom2nifti Docker container using the `docker run` command. This command runs the Docker image with the specified input files:

    ```sh
    sudo docker run -it --rm --name TF2.4_IVIM-MRI_CodeCollection \
        -v ~/TF2.4_IVIM-MRI_CodeCollection:/usr/src/app \
        -v ~/TF2.4_IVIM-MRI_CodeCollection:/usr/app/output \
        tf2.4_ivim-mri_codecollection \
        /usr/src/app/dicom_folder /usr/app/output
    ```

- You can also run the command in an interactive mode if you want to convert multiple folders of data and run them consecutively.

    ```sh
    python ./WrapImage/dicom2niix_wrapper.py --prompt-user
    ```

   It then prompts for inquiring information of DICOM Images in your terminal.

   The interactive version of this script is not suited for use in container.

[Note that NIfTI and DICOM encode space differently](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#Spatial_Coordinates)

![image](https://github.com/user-attachments/assets/8ea21692-36ac-4773-aec7-6cb3a6838055)

##### The goal of dcm2niix is to create FSL format bvec/bval files for processing. A crucial concern is ensuring that the gradient directions are reported in the frame of reference expected by the software you use to fit your tractography. [dicom2niix should generate a ".bvec" file that reports the tensors as expected](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#Diffusion_Tensor_Imaging) by FSL's dtifit, where vectors are reported relative to image frame of reference (rather than relative to the scanner bore). 

#### It is strongly recommend that users check validate the b-vector directions for their hardware and sequence as [described in a dedicated document](https://www.nitrc.org/docman/?group_id=880)
---
