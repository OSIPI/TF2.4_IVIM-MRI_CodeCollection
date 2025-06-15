FROM debian:stable-slim AS build

ARG DCM2NIIX_VERSION=v1.0.20241211

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libssl-dev \
    wget \
    pigz \
    ca-certificates \
    && update-ca-certificates \
    && wget https://github.com/rordenlab/dcm2niix/archive/refs/tags/${DCM2NIIX_VERSION}.tar.gz -O /tmp/dcm2niix.tar.gz \
    && mkdir -p /tmp/dcm2niix && tar -xzf /tmp/dcm2niix.tar.gz -C /tmp/dcm2niix --strip-components=1 \
    && mkdir /tmp/dcm2niix/build && cd /tmp/dcm2niix/build \
    && cmake -DBATCH_VERSION=ON -DZLIB_IMPLEMENTATION=Cloudflare -DUSE_JPEGLS=ON -DUSE_OPENJPEG=ON .. \
    && make && make install \
    && rm -rf /tmp/dcm2niix* \
    && apt-get remove -y wget git cmake \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

FROM python:3.11-slim

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    pigz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/dcm2niix /usr/local/bin/dcm2niix

COPY ../.. .

RUN pip install --no-cache-dir .

ENTRYPOINT ["python3", "-m", "WrapImage.dicom2niix_wrapper"]
