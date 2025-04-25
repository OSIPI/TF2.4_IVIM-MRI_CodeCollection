import argparse
import io
import os
from pathlib import Path
import selectors
import subprocess
import sys
import inquirer
import itk
import numpy as np
import datetime
import uuid

def save_dicom_objects(
    volume_4d: np.ndarray,
    bvals: list,
    bvecs: list,
    out_dir: Path,
    f_vals: np.ndarray = None,
    D_vals: np.ndarray = None,
    Dp_vals: np.ndarray = None,
    patient_id: str = "SIM123",
    modality: str = "MR",
    pixel_spacing=(1.0, 1.0),
    slice_thickness=1.0,
    series_description="Simulated IVIM",
    protocol_name="SimulatedProtocol",
    patient_position="HFS",
    manufacturer="Simulated Manufacturer",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, z, t = volume_4d.shape
    assert t == len(bvals) == len(bvecs), "Mismatch in timepoints and bvals/bvecs"

    # Define 2D image type (DICOM expects 2D slices)
    ImageType = itk.Image[itk.SS, 2]  # 2D image with short (int16) pixel type
    WriterType = itk.ImageFileWriter[ImageType]

    for t_idx in range(t):
        volume_3d = volume_4d[:, :, :, t_idx].astype(np.int16)

        # Generate common metadata for this timepoint
        study_uid = f"1.2.826.0.1.3680043.2.1125.{uuid.uuid4().int >> 64}"
        series_uid = f"1.2.826.0.1.3680043.2.1125.{uuid.uuid4().int >> 64}"
        now = datetime.datetime.now().strftime("%Y%m%d")
        series_number = str(t_idx + 1)

        for slice_idx in range(z):
            # Create 2D slice
            slice_2d = volume_3d[:, :, slice_idx]
            image_2d = itk.image_view_from_array(slice_2d.T)  # Transpose for proper orientation
            image_2d = itk.cast_image_filter(image_2d, ttype=(type(image_2d), ImageType))

            # Set image properties
            image_2d.SetSpacing(pixel_spacing)
            image_2d.SetOrigin([0.0, 0.0])

            # Create metadata dictionary
            meta_dict = itk.MetaDataDictionary()
            meta_dict["0010|0010"] = "Simulated^Patient"
            meta_dict["0010|0020"] = patient_id
            meta_dict["0008|0060"] = modality
            meta_dict["0008|0020"] = now
            meta_dict["0008|0030"] = "120000"
            meta_dict["0020|000d"] = study_uid
            meta_dict["0020|000e"] = series_uid
            meta_dict["0020|0011"] = series_number
            meta_dict["0008|103e"] = f"{series_description} Volume {series_number}"
            meta_dict["0018|1030"] = protocol_name
            meta_dict["0018|5100"] = patient_position 
            meta_dict["0008|0070"] = manufacturer

            meta_dict["0018|9087"] = str(float(bvals[t_idx]))
            gx, gy, gz = bvecs[t_idx]
            meta_dict["0018|9089"] = f"{gx}\\{gy}\\{gz}"

            if f_vals is not None:
                meta_dict["0011|1001"] = f"f_in_mean={np.mean(f_vals):.6f}"
            if D_vals is not None:
                meta_dict["0011|1002"] = f"D_in_mean={np.mean(D_vals):.6e}"
            if Dp_vals is not None:
                meta_dict["0011|1003"] = f"Dp_in_mean={np.mean(Dp_vals):.6e}"

            sop_uid = f"1.2.826.0.1.3680043.8.498.{uuid.uuid4().int >> 64}"
            meta_dict["0008|0018"] = sop_uid
            meta_dict["0020|0013"] = str(slice_idx + 1)
            meta_dict["0020|0032"] = f"0\\0\\{slice_idx * slice_thickness}"
            meta_dict["0020|0037"] = "1\\0\\0\\0\\1\\0"

            # Create and configure writer
            writer = WriterType.New()
            writer.SetInput(image_2d)
            writer.SetFileName(str(out_dir / f"vol_{t_idx:03d}_slice_{slice_idx:03d}.dcm"))

            # Use GDCMImageIO for DICOM writing
            gdcm_io = itk.GDCMImageIO.New()
            gdcm_io.SetMetaDataDictionary(meta_dict)
            gdcm_io.KeepOriginalUIDOn()

            writer.SetImageIO(gdcm_io)
            writer.Update()

    print(f" DICOMs written to {out_dir.resolve()}")

def prompt_input_directory():
    return inquirer.prompt([
        inquirer.Path(
            "path",
            message="📂 Select an input directory containing DICOM image files:",
            path_type=inquirer.Path.DIRECTORY,
            exists=True,
        )
    ])["path"]


def prompt_output_directory(input_dir):
    # List subfolders of input_dir
    subdirs = [
        name for name in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, name))
    ]

    choices = [f"./{name}" for name in subdirs]
    choices.append("📥 Enter a custom output path...")

    answer = inquirer.prompt([
        inquirer.List(
            "choice",
            message=f"📁 Choose an output directory for NIfTI files from:\n → {input_dir}",
            choices=choices
        )
    ])["choice"]

    if answer == "📥 Enter a custom output path...":
        return inquirer.prompt([
            inquirer.Path(
                "custom_path",
                message="📥 Enter custom output directory path:",
                path_type=inquirer.Path.DIRECTORY,
                exists=True
            )
        ])["custom_path"]
    else:
        return os.path.abspath(os.path.join(input_dir, answer.strip("./")))


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

    selector = selectors.DefaultSelector()               # register callback 
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

def run_interactive():

    input_dirs = []
    output_dirs = []

    while True:
        input_dir = prompt_input_directory()
        output_dir = prompt_output_directory(input_dir)

        input_dirs.append(input_dir)
        output_dirs.append(output_dir)

        add_more = inquirer.prompt([
            inquirer.Confirm("more", message="➕ Add another input/output pair?", default=False)
        ])["more"]

        if not add_more:
            break

    merge_answer = inquirer.prompt([
        inquirer.Confirm("merge", message="🧩 Merge 2D slices into a single NIfTI (-m y)?", default=True)
    ])
    merge_2d = merge_answer["merge"]

    single_file = inquirer.prompt([
        inquirer.Confirm("single", message="📦 Force single file input (-s y)?", default=False)
    ])["single"]

    for in_dir, out_dir in zip(input_dirs, output_dirs):
        vol_dir = Path(in_dir)
        out_path = Path(out_dir)

        print(f"Converting:\n → Input: {vol_dir}\n → Output: {out_path}")
        try:
            nifti, bval, bvec = dicom_to_niix(vol_dir, out_path, merge_2d, single_file)
            print(f" Conversion succeeded: {nifti}")
        except RuntimeError as err:
            print(f"❌ Conversion failed: {err}")

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
    parser.add_argument("-pu", "--prompt-user", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    if args.prompt_user:
        run_interactive()
    elif args.input:
        run_cli(args.input, args.output, **vars(args))
    else:
        print("❗ You must provide input and output paths OR use --prompt-user for interactive mode.")
        parser.print_help()
        sys.exit(1)