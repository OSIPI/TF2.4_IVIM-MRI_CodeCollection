# IVIM Fitting Method Workflow - Technical Documentation

## Overview

The **IVIM (Intravoxel Incoherent Motion) Fitting Method Workflow** is a comprehensive Kaapana-based pipeline for processing medical imaging data. This workflow supports both DICOM and NIfTI input formats, performs IVIM model fitting using multiple algorithms, and outputs parametric maps for medical analysis.

### Key Features

- **Multi-format support**: Handles both DICOM and NIfTI input data
- **Algorithm flexibility**: Supports 18+ different IVIM fitting algorithms
- **Containerized deployment**: Uses Docker containers orchestrated through Airflow DAGs
- **Scalable architecture**: Deployed via Helm charts in Kubernetes environments
- **Parameter extraction**: Generates f, Dp, and D parametric maps from diffusion-weighted imaging

## Architecture Overview

The IVIM Fitting Method Workflow supports dual input paths with branching logic based on upload type. DICOM inputs undergo conversion to NIfTI format before IVIM fitting, while NIfTI inputs proceed directly to fitting. Both paths converge at the IVIM fitting stage and produce identical parametric map outputs.

## Directory Structure

```
.
├── extension/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── files/
│   │       ├── dag_ivim-fitting_method.py
│   │       └── ivim-fitting-method/
│   │           ├── DicomToNiftiOperator.py
│   │           └── MyIvimFittingOperator.py
│   └── ivim-fitting-method-workflow/
│       ├── Chart.yaml
│       ├── requirements.yaml
│       └── values.yaml
└── processing-containers/
    ├── dicom_to_nifti/
    │   ├── Dockerfile
    │   ├── convert_dicom_to_nifti.py
    │   └── requirements.txt
    └── ivim-fitting-method/
        ├── Dockerfile
        └── nifti_wrapper_kaapana.py
```

## Workflow Components

### 1. DAG Definition (`dag_ivim-fitting_method.py`)

The main orchestration file defining the Airflow DAG with the following key features:

#### UI Configuration

The workflow form includes key parameters:

- `upload_type`: Determines processing path ("nifti" or "dicom")
- `source_files`: Specifies comma-separated file paths for MinIO download (NIfTI, bval, bvec files)
- `algorithm`: Selects from 18+ available IVIM fitting algorithms (default: "OJ_GU_seg")
- `algorithm_args`: Optional additional parameters for algorithms
- `affine`: Optional affine matrix override

The `source_files` parameter is critical for the NIfTI workflow path as it specifies which files the MinIO get operator should download from the `uploads/Data/` prefix. The files are specified as comma-separated paths and must include the NIfTI image, b-values file, and b-vectors file in that order.

#### Branching Logic

The workflow uses a `BranchPythonOperator` to determine the processing path based on input type:

```python
def choose_upload_type(**context):
    upload_type = context["dag_run"].conf.get("workflow_form", {}).get("upload_type", "nifti")
    if upload_type == "dicom":
        return "get_input"
    return "download-data"
```

#### Task Dependencies

```python
# DICOM Processing Path
branch_task >> get_input >> dicom_to_nifti >> ivim_fitting_task_dicom >> put_output_to_minio_dicom

# NIfTI Processing Path
branch_task >> get_data >> ivim_fitting_task_nifti >> put_output_to_minio_nifti

# Cleanup
[put_output_to_minio_nifti, put_output_to_minio_dicom] >> clean
```

### 2. Custom Operators

#### DicomToNiftiOperator

Extends KaapanaBaseOperator with specific configuration for DICOM-to-NiFTI conversion. Sets execution timeout, memory allocation, and references the dicom-to-nifti container image.

#### MyIvimFittingOperator

Extends KaapanaBaseOperator for IVIM model fitting execution. Configures resource limits and references the ivim-fitting-method container image.

### Helper Functions in `nifti_wrapper_kaapana.py`

The implementation includes these helper functions:

- `read_nifti_file()`: Loads NIfTI file and returns data, affine matrix, and header
- `read_bval_file()`: Reads b-values from file as float array
- `read_bvec_file()`: Reads and transposes b-vectors for correct orientation
- `save_nifti_file()`: Saves data as NIfTI file with given affine matrix
- `loop_over_first_n_minus_1_dimensions()`: Generator for voxel-wise iteration
- `load_config()`: Loads workflow configuration from JSON file

#### Input Data Handling

The implementation handles both NIfTI and DICOM inputs based on the upload type configuration:

- **NIfTI path**: Parses comma-separated file paths from UI form and resolves to input directory
- **DICOM path**: Searches batch folders for converted NIfTI files with associated bval/bvec files

#### IVIM Model Fitting Implementation

The IVIM fitting process:

1. Loads data using helper functions (`read_nifti_file`, `read_bval_file`, `read_bvec_file`)
2. Initializes OsipiBase with selected algorithm
3. Preallocates output arrays for f, Dp, and D parameters
4. Iterates voxel-by-voxel using `loop_over_first_n_minus_1_dimensions`
5. Fits IVIM model and extracts parameters for each voxel

#### Output Management

Results are saved as separate NIfTI files for each parameter map using `save_nifti_file`. Original input files are copied to the workflow directory as a Kaapana requirement.

## Supported IVIM Algorithms

The workflow supports 18 different IVIM fitting algorithms from the OSIPI (Open Source Initiative for Perfusion Imaging) standardization effort:

### Biexponential Fitting Methods

- **ASD_MemorialSloanKettering_QAMPER_IVIM**: Quality assessment and parameter estimation
- **IAR_LU_biexp**: Standard biexponential fitting
- **OGC_AmsterdamUMC_biexp**: Amsterdam UMC implementation
- **PV_MUMC_biexp**: Maastricht University implementation

### Segmented Fitting Methods

- **IAR_LU_segmented_2step**: Two-step segmented approach
- **IAR_LU_segmented_3step**: Three-step segmented approach
- **OGC_AmsterdamUMC_biexp_segmented**: Segmented biexponential

### Linear and Advanced Methods

- **ETP_SRI_LinearFitting**: Linear regression approach
- **TCML_TechnionIIT_SLS**: Sequential linear squares
- **TCML_TechnionIIT_lsqBOBYQA**: Bound optimization method
- **OGC_AmsterdamUMC_Bayesian_biexp**: Bayesian inference approach

### Specialized Methods

- **IAR_LU_modified_mix**: Modified mixture model
- **IAR_LU_modified_topopro**: Topology-preserving modification
- **IAR_LU_subtracted**: Background subtraction method
- **OJ_GU_seg**: Segmentation-based fitting (default)
- **PvH_KB_NKI_IVIMfit**: Netherlands Cancer Institute implementation

## Deployment Configuration

### Helm Chart (`Chart.yaml`)

Defines metadata for the Helm chart deployment including name, version (0.5.1-5-g1a6aae746), application version (0.1.0), and maintainer information. Includes keywords for workflow identification and references to OSIPI Task Force 2.4.

### Dependencies (`requirements.yaml`)

Specifies Helm chart dependencies required by the workflow. Declares a dependency on `dag-installer-chart` with local file path reference to the Kaapana services directory where the DAG installer chart is located.

### Configuration (`values.yaml`)

```yaml
global:
  image: "dag-ivim-fitting-method"
  action: "copy"
```

## Container Architecture

### DAG Container (`extension/docker/Dockerfile`)

Packages DAG definition and custom operators for Kaapana deployment. Contains the workflow orchestration files and operator classes.

### IVIM Fitting Container (`processing-containers/ivim-fitting-method/Dockerfile`)

Based on local-only/base-python-cpu image. Installs system dependencies including build tools and LaTeX packages, clones the OSIPI IVIM code collection from GitHub, installs Python requirements, and sets `nifti_wrapper_kaapana.py` as the entry point module.

**Purpose**: Executes the IVIM fitting logic by cloning the OSIPI IVIM code collection and running the `nifti_wrapper_kaapana.py` module as the entry point. The container installs necessary system dependencies and Python packages required for IVIM analysis.

### DICOM to NIfTI Container

Located in `processing-containers/dicom_to_nifti/`. Converts DICOM series to NIfTI format with gradient table extraction.

## IVIM Parameter Extraction

The workflow extracts three key IVIM parameters:

- **f (Perfusion Fraction)**: Fraction of signal from flowing blood
- **Dp (Pseudo-diffusion Coefficient)**: Perfusion-related diffusion coefficient
- **D (Diffusion Coefficient)**: True tissue diffusion coefficient

## Troubleshooting

### Common Issues

#### 1. Memory Errors

```python
# Increase operator memory limits
ram_mem_mb=8000,
ram_mem_mb_lmt=16000
```

#### 2. File Path Issues

```python
# Verify input file paths
print(f"Looking for files in: {element_input_dir}")
print(f"Available files: {os.listdir(element_input_dir)}")
```

#### 3. Algorithm Failures

```python
# Check algorithm parameters
if algorithm_args:
    print(f"Using algorithm arguments: {algorithm_args}")
```

### Debugging Steps

1. Check Airflow logs for task failures
2. Verify input file formats and integrity
3. Monitor container resource usage
4. Validate algorithm parameter compatibility
5. Review MinIO storage permissions

## Troubleshooting

### Common Issues

#### Memory Errors

Increase operator memory limits in operator initialization parameters.

#### File Path Issues

Verify input file paths and check available files in input directories.

#### Algorithm Failures

Validate algorithm parameters and check compatibility with input data.

### Debugging Steps

1. Check Airflow logs for task failures
2. Verify input file formats and integrity
3. Monitor container resource usage
4. Validate algorithm parameter compatibility
5. Review MinIO storage permissions

## References and Resources

- **OSIPI Task Force 2.4**: [IVIM Standardization](https://osipi.ismrm.org/task-forces/task-force-2-4/)
- **Kaapana Documentation**: [Kaapana Framework](https://kaapana.readthedocs.io/)
