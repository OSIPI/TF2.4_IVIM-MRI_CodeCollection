from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.models import DAG
from airflow.operators.python import BranchPythonOperator

# Kaapana operators
from kaapana.operators.MinioOperator import MinioOperator
from kaapana.operators.LocalWorkflowCleanerOperator import LocalWorkflowCleanerOperator
from kaapana.operators.GetInputOperator import GetInputOperator
from kaapana.operators.DcmConverterOperator import DcmConverterOperator

# IVIM custom operator
from ivimfitting_method.MyIvimFittingOperator import MyIvimFittingOperator
from ivimfitting_method.DicomToNiftiOperator import DicomToNiftiOperator


# -----------------------------
# UI forms
# -----------------------------
ui_forms = {
    "workflow_form": {
        "type": "object",
        "properties": {
            "upload_type": {
                "title": "Upload Type",
                "description": "Choose whether you are uploading NIfTI (with bval/bvec) or raw DICOMs.",
                "type": "string",
                "enum": ["nifti", "dicom"],
                "default": "nifti",
                "readOnly": False
            },
            "source_files": {
                "title": "Comma-separated paths to NIfTI, B-vector, and B-value files",
                "description": "List containing: Path to the input 4D NIfTI file, Path to the b-vector file, and Path to the b-value file, separated by commas. Example: brain.nii.gz,brain.bval,brain.bvec.",
                "type": "string",
                "readOnly": False
            },
            "affine": {
                "title": "Affine Matrix",
                "description": "Affine matrix for NIfTI image. Enter space-separated numbers (e.g. 1 0 0 0 0 1 0 0 0 0 1 0).",
                "type": "array",
                "items": {
                    "type": "number"
                },
                "readOnly": False,
            },
            "algorithm": {
                "title": "Segmentation Algorithm",
                "description": "Select the algorithm to use.",
                "type": "string",
                "enum": [
                    "ASD_MemorialSloanKettering_QAMPER_IVIM",
                    "ETP_SRI_LinearFitting",
                    "IAR_LU_biexp",
                    "IAR_LU_modified_mix",
                    "IAR_LU_modified_topopro",
                    "IAR_LU_segmented_2step",
                    "IAR_LU_segmented_3step",
                    "IAR_LU_subtracted",
                    "OGC_AmsterdamUMC_Bayesian_biexp",
                    "OGC_AmsterdamUMC_biexp",
                    "OGC_AmsterdamUMC_biexp_segmented",
                    "OJ_GU_seg",
                    "PV_MUMC_biexp",
                    "PvH_KB_NKI_IVIMfit",
                    "TCML_TechnionIIT_SLS",
                    "TCML_TechnionIIT_lsqBOBYQA",
                    "TCML_TechnionIIT_lsqlm",
                    "TCML_TechnionIIT_lsqtrf"
                ],
                "default": "OJ_GU_seg",
                "readOnly": False
            },
            "algorithm_args": {
                "title": "Additional Algorithm Arguments",
                "description": "Optional extra arguments passed to the algorithm.",
                "type": "string",
                "readOnly": False,
            },
        },
    }
}

log = LoggingMixin().log


# -----------------------------
# Default DAG args
# -----------------------------
args = {
    "owner": "a",
    "start_date": days_ago(0),
    "retries": 0,
    "retry_delay": timedelta(seconds=30),
    "ui_visible": True,
    "ui_forms": ui_forms
}


# -----------------------------
# DAG definition
# -----------------------------
dag = DAG(
    dag_id="ivim-fitting-method",
    default_args=args,
    schedule_interval=None
)


# -----------------------------
# Branch function
# -----------------------------
def choose_upload_type(**context):
    wf_form = context["dag_run"].conf.get("workflow_form", {})
    upload_type = wf_form.get("upload_type", "nifti")
    log.info(f"Upload type selected: {upload_type}")
    if upload_type == "dicom":
        return "get_input"
    return "download-data"


branch_task = BranchPythonOperator(
    task_id="choose-upload-type",
    python_callable=choose_upload_type,
    dag=dag,
)


# -----------------------------
# DICOM path
# -----------------------------
get_input = GetInputOperator(
    dag=dag,
    name="get_input"
)

dicom_to_nifti = DicomToNiftiOperator(
    dag=dag,
    name="dicom_to_nifti",
    input_operator=get_input,
)

# -----------------------------
# NIfTI path
# -----------------------------
get_data = MinioOperator(
    dag=dag,
    name="download-data",
    minio_prefix="uploads/Data/",
    action="get",
)


# -----------------------------
# Shared IVIM fitting
# -----------------------------
ivim_fitting_task_dicom = MyIvimFittingOperator(
    dag=dag,
    name="ivim_fitting_task_dicom",
    input_operator=dicom_to_nifti
)

ivim_fitting_task_nifti = MyIvimFittingOperator(
    dag=dag,
    name="ivim_fitting_task_nifti",
    input_operator=get_data
)


# -----------------------------
# Output + cleanup
# -----------------------------
put_output_to_minio_nifti = MinioOperator(
    dag=dag,
    name="upload-out-nifti",
    minio_prefix="uploads",
    action="put",
    none_batch_input_operators=[ivim_fitting_task_nifti],
    whitelisted_file_extensions=(".nii.gz",),
)

put_output_to_minio_dicom = MinioOperator(
    dag=dag,
    name="upload-out-dicom",
    minio_prefix="uploads",
    action="put",
    none_batch_input_operators=[ivim_fitting_task_dicom],
    whitelisted_file_extensions=(".nii.gz",),
)

clean = LocalWorkflowCleanerOperator(
    dag=dag,
    clean_workflow_dir=True,
    trigger_rule="none_failed_or_skipped"
)


# -----------------------------
# Task dependencies
# -----------------------------
branch_task >> [get_input, get_data]

get_input >> dicom_to_nifti >> ivim_fitting_task_dicom >> put_output_to_minio_dicom
get_data >> ivim_fitting_task_nifti >> put_output_to_minio_nifti

put_output_to_minio_nifti >> clean
put_output_to_minio_dicom >> clean
