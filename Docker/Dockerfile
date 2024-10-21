FROM python:3.11-slim

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ../requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY .. .

ENTRYPOINT ["python3", "-m", "WrapImage.nifti_wrapper"]
