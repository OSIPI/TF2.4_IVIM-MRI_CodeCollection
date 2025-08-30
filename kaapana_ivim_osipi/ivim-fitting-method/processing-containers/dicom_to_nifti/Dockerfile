FROM python:3.12-slim

LABEL IMAGE="dicom-to-nifti"
LABEL VERSION="0.1.0"
LABEL BUILD_IGNORE="False"

WORKDIR /kaapana/app

# Install dependencies: dcm2niix + dcmtk (for dcmdjpls)
RUN apt-get update && apt-get install -y --no-install-recommends \
    dcm2niix \
    dcmtk \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy script
COPY convert_dicom_to_nifti.py ./

ENTRYPOINT ["python3", "-u", "convert_dicom_to_nifti.py"]

