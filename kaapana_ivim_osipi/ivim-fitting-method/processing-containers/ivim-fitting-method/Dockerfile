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
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/OSIPI/TF2.4_IVIM-MRI_CodeCollection.git

WORKDIR /kaapana/app/TF2.4_IVIM-MRI_CodeCollection
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "-u", "-m", "WrapImage.nifti_wrapper_kaapana"]
