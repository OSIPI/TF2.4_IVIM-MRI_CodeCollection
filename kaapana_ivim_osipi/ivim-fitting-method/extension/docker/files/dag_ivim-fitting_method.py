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
            "single_execution": {
                "title": "single execution",
                "description": "Should each series be processed separately?",
                "type": "boolean",
                "default": False,
                "readOnly": False,
            }
        },
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
    schedule_interval=None)

get_data = MinioOperator(
    dag=dag,
    name="download-data",
    minio_prefix="uploads/Data/",
    action="get",
    source_files=["brain.bvec", "brain.bval", "brain.nii.gz"]
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

get_data >> ivim_fitting_task >> put_output_to_minio >> clean
