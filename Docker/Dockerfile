FROM python:3.11-alpine

WORKDIR /usr/src/app

# Install build tools and necessary libraries on Alpine
RUN apk add --no-cache \
    build-base \
    libssl1.1 \
    openssl-dev

# Copy and install Python dependencies
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY .. .

# Define the entry point
ENTRYPOINT ["python3", "-m", "WrapImage.nifti_wrapper"]
