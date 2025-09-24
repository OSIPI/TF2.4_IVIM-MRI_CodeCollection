#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from glob import glob
from tqdm import tqdm

WORKFLOW_DIR = os.environ["WORKFLOW_DIR"]
OPERATOR_IN_DIR = os.environ.get("OPERATOR_IN_DIR", "get_input")
OPERATOR_OUT_DIR = os.environ.get("OPERATOR_OUT_DIR", "dicom_to_nifti")

BATCH_DIR = os.path.join(WORKFLOW_DIR, "batch")

# Loop over each batch element (usually one per patient/series)
batch_folders = sorted([f for f in glob(os.path.join(BATCH_DIR, "*"))])
print(f"batch-folders - {batch_folders}")

for batch_element_dir in tqdm(batch_folders, desc="Processing batch elements"):
    element_input_dir = os.path.join(batch_element_dir, OPERATOR_IN_DIR)
    element_output_dir = os.path.join(batch_element_dir, OPERATOR_OUT_DIR)

    if not os.path.exists(element_input_dir):
        print(f"# Input-dir {element_input_dir} does not exist. Skipping...")
        continue

    Path(element_output_dir).mkdir(parents=True, exist_ok=True)

    print(f" Converting {element_input_dir} → {element_output_dir}")

    # Run dcm2niix directly on the input folder
    cmd = [
        "dcm2niix",
        "-z", "y",               # compress output (gz)
        "-f", "%p_%t_%s",        # output filename pattern
        "-o", element_output_dir,
        element_input_dir
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"# Conversion completed for {element_input_dir}")
    except subprocess.CalledProcessError:
        print(f"# ERROR: dcm2niix failed for {element_input_dir}")
