FROM python:3.12-slim

LABEL IMAGE="dicom-to-nifti"
LABEL VERSION="0.1.0"
LABEL BUILD_IGNORE="False"

WORKDIR /kaapana/app

# Install dependencies: curl, unzip, build tools (needed by Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install latest dcm2niix (from GitHub release)
RUN curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip \
    && unzip dcm2niix_lnx.zip -d /usr/local/bin \
    && rm dcm2niix_lnx.zip

# Copy requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy script
COPY convert_dicom_to_nifti.py ./

ENTRYPOINT ["python3", "-u", "convert_dicom_to_nifti.py"]
