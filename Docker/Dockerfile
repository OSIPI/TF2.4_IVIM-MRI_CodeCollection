FROM python:3.11-slim

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    wget \
    pigz \
    && wget https://github.com/rordenlab/dcm2niix/archive/refs/tags/v1.0.20241211.tar.gz \
    && tar -xzf v1.0.20241211.tar.gz -C /usr/local/bin \
    && rm v1.0.20241211.tar.gz \
    && apt-get remove -y wget \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ../requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY .. .

ENTRYPOINT ["python3", "-m", "WrapImage.nifti_wrapper"]
