#!/usr/bin/env python3
import os
import subprocess
import shutil
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

    # Create a temporary raw_dicoms directory
    raw_dicoms_dir = os.path.join(batch_element_dir, "raw_dicoms")
    Path(raw_dicoms_dir).mkdir(parents=True, exist_ok=True)

    print(f" Decoding JPEG-LS DICOMs from {element_input_dir} → {raw_dicoms_dir}")

    # Decode each DICOM using dcmdjpls
    dcm_files = glob(os.path.join(element_input_dir, "*.dcm"))
    for f in tqdm(dcm_files, desc="Decoding JPEG-LS", leave=False):
        out_file = os.path.join(raw_dicoms_dir, os.path.basename(f))
        try:
            subprocess.run(["dcmdjpls", f, out_file], check=True)
        except subprocess.CalledProcessError:
            print(f"# ERROR: Failed to decode {f} with dcmdjpls")

    # Run dcm2niix on the decoded folder
    print(f" Converting raw_dicoms → {element_output_dir}")
    cmd = [
        "dcm2niix",
        "-z", "y",               # compress output
        "-f", "%p_%t_%s",        # output filename pattern
        "-o", element_output_dir,
        raw_dicoms_dir
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"# Conversion completed for {element_input_dir}")
    except subprocess.CalledProcessError:
        print(f"# ERROR: dcm2niix failed for {element_input_dir}")
    finally:
        # Clean up raw_dicoms folder
        if os.path.exists(raw_dicoms_dir):
            shutil.rmtree(raw_dicoms_dir)
            print(f"# Deleted temporary directory {raw_dicoms_dir}")

