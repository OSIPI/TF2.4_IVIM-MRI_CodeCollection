from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.models import DAG
from ivimfitting_method.MyIvimFittingOperator import MyIvimFittingOperator
from kaapana.operators.MinioOperator import MinioOperator
from kaapana.operators.LocalWorkflowCleanerOperator import LocalWorkflowCleanerOperator

ui_forms = {
    "workflow_form": {
        "type": "object",
        "properties": {
            "source_files": {
                "title": "Comma-separated paths to NIfTI, B-vector, and B-value files",
                "description": "List containing: Path to the input 4D NIfTI file, Path to the b-vector file, and Path to the b-value file, separated by commas. Example: brain.nii.gz,brain.bval,brain.bvec. These are the files to download from MinIO relative to the bucket name and minio_prefix.",
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
        "required": ["source_files"]
    }
}

log = LoggingMixin().log


# Default DAG args
args = {
    "owner": "a",
    "start_date": days_ago(0),
    "retries": 0,
    "retry_delay": timedelta(seconds=30),
    "ui_visible": True,
    "ui_forms": ui_forms
}

# Create the DAG
dag = DAG(
    dag_id="ivim-fitting-method",
    default_args=args,
    schedule_interval=None
)

# Fixed variable name to match usage
get_data = MinioOperator(
    dag=dag,
    name="download-data",
    minio_prefix="uploads/Data/",
    action="get",
)

# Define your operator task
ivim_fitting_task = MyIvimFittingOperator(
    dag=dag,
    input_operator=get_data
)

put_output_to_minio = MinioOperator(
    dag=dag,
    name="upload-out",
    minio_prefix="uploads",
    action="put",
    none_batch_input_operators=[ivim_fitting_task],
    whitelisted_file_extensions=(".nii.gz",),
)

clean = LocalWorkflowCleanerOperator(dag=dag, clean_workflow_dir=True)

# Task dependencies
get_data >> ivim_fitting_task >> put_output_to_minio >> clean

