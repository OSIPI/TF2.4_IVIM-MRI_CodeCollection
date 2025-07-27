FROM local-only/base-python-cpu:latest

LABEL IMAGE="ivim-fitting-method"
LABEL VERSION="0.1.0"
LABEL BUILD_IGNORE="False"

WORKDIR /kaapana/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY TF2.4_IVIM-MRI_CodeCollection/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY TF2.4_IVIM-MRI_CodeCollection/ .

ENTRYPOINT ["python3", "-u", "-m", "WrapImage.nifti_wrapper_kaapana"]
