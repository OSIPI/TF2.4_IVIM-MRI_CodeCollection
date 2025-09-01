FROM local-only/base-installer:latest

LABEL IMAGE="dag-ivim-fitting-method"
LABEL VERSION="0.1.0"
LABEL BUILD_IGNORE="False"

COPY files/dag_ivim-fitting_method.py /kaapana/tmp/dags/
COPY files/ivim-fitting-method/MyIvimFittingOperator.py /kaapana/tmp/dags/ivimfitting_method/
COPY files/ivim-fitting-method/DicomToNiftiOperator.py /kaapana/tmp/dags/ivimfitting_method/
