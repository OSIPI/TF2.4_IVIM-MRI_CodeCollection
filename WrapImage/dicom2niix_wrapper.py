import argparse
import io
import os
from pathlib import Path
import selectors
import subprocess
import sys
import numpy as np



def dicom_to_niix(vol_dir: Path, out_dir: Path = None, merge_2d: bool = False, is_single_file: bool = False):
    """
    For converting DICOM images to a (compresssed) 4d nifti image
    """
    if not out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "dcm2niix",
        "-f", "%s_%p", # dcm2niix attempts to provide a sensible file naming scheme
        "-o", out_dir if out_dir else "",  # Add merge option
        "-z", "y", # parallel pigz compressed nii.gz file; 'optimal'"-z o" which pipes data directly to pigz
        "-m", "y" if merge_2d else "n",  # Add merge option
        "-s", "y" if is_single_file else "n",
        # https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage
        # for further configuration for general usage see page above
        vol_dir
    ]

    try:
        success, output = capture_subprocess_output(cmd)
        print(output)
        if not success:
            raise RuntimeError(f"dcm2niix failed: {output}")
        
        nifti_files = list(Path(out_dir).glob("*.nii.gz"))

        if is_single_file and len(nifti_files) != 1:
            raise RuntimeError("Expected a single .nii.gz output due to flags"
                               " Please collect your  NIfTI files in the output directory prior to execution.")
        if merge_2d and len(nifti_files) != 1:
            raise RuntimeError("Expected a single .nii.gz output due to flags"
                               " Check the Warnings logged by dicom2niix to see source of error.")
        if len(nifti_files) < 1:
            raise RuntimeError("No NIfTI (.nii.gz) files were generated."
                               " Check the Warnings logged by dicom2niix and Double-check your input before running again.")
            
        bval_files = list(out_dir.glob("*.bval"))
        bvec_files = list(out_dir.glob("*.bvec"))
        bval_path = str(bval_files[0]) if bval_files else None
        bvec_path = str(bvec_files[0]) if bvec_files else None

        if not bval_path or not bvec_path:
            raise RuntimeError("No bvec or bval files were generated.")

        return nifti_files[0], bval_path, bvec_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"dcm2niix failed: {e.stderr}")
    
### Adapted from https://gist.github.com/nawatts/e2cdca610463200c12eac2a14efc0bfb ###
### for further breakdown and see gist above ###
### for future improvements https://me.micahrl.com/blog/magicrun/ to be called over SSH OR SLURM. ###
def capture_subprocess_output(subprocess_args):
    process = subprocess.Popen(
        subprocess_args,
        bufsize=1, # output is line buffered
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True # for line buffering
    )

    buf = io.StringIO() # callback for output
    def handle_output(stream, mask):
        line = stream.readline()
        buf.write(line)
        sys.stdout.write(line)

    selector = selectors.DefaultSelector() # register callback 
    selector.register(process.stdout, selectors.EVENT_READ, handle_output) # for 'read' event from subprocess stdout stream

    while process.poll() is None:
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    # ensure all remaining output is processed
    while True:
        line = process.stdout.readline()
        if not line:
            break
        buf.write(line)
        sys.stdout.write(line)

    return_code = process.wait()
    selector.close()

    success = (return_code == 0)
    output = buf.getvalue()
    buf.close()

    return success, output

def run_cli(input_path: str, output_path: str, **kwargs):
    vol_dir = Path(input_path)
    out_dir = Path(output_path) if output_path else vol_dir

    merge_2d = kwargs.get("merge_2d", False)
    single_file = kwargs.get("single_file", False)

    print(f" Converting:\n → Input: {vol_dir}\n → Output: {out_dir}")
    try:
        nifti, bval, bvec = dicom_to_niix(vol_dir, out_dir, merge_2d, single_file)
        print(f" Created NIfTI: {nifti}")

        if bval and bvec:
            print(" Running IVIM fitting algorithm...")
            subprocess.run([
                "python3", "-m", "WrapImage.nifti_wrapper",
                str(nifti), str(bvec), str(bval)
            ], check=True)
            print(" IVIM fitting complete.")
        else:
            print("⚠️ bvec/bval missing, skipping IVIM post-processing.")
    except RuntimeError as err:
        print(f"❌ Conversion failed: {err}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DICOM to NIfTI converter with optional IVIM processing")
    parser.add_argument("input", nargs="?", help="Path to input DICOM directory")
    parser.add_argument("-o", "--output", nargs="?", help="Path to output directory for NIfTI files, defaults to the same folder as the original DICOM files.")
    parser.add_argument("-m", "--merge-2d", action="store_true", help="Merge 2D slices (-m y)")
    parser.add_argument("-s", "--single-file", action="store_true", help="Enable single file mode (-s y)")

    args = parser.parse_args()

    if args.input:
        run_cli(args.input, args.output, **vars(args))
    else:
        print("❗ You must provide input and output paths ")
        parser.print_help()
        sys.exit(1)