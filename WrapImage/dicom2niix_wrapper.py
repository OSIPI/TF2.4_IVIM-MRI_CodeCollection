import argparse
import os
from pathlib import Path
import subprocess
import sys
import inquirer

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


def dicom_to_niix(vol_dir: Path, out_dir: Path, merge_2d: bool = False, series_num: int = -1, is_single_file: bool = False):
    """
    For converting DICOM images to a (compresssed) 4d nifti image
    """

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "dcm2niix",
        "-f", "%s_%p", # dcm2niix attempts to provide a sensible file naming scheme
        "-o", out_dir, # output directory
        "-z", "y", #specifying compressed nii.gz file
        "-m", "y" if merge_2d else "n",  # Add merge option
        "-s", "y" if is_single_file else "n",
        # https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage
        # for further configuration for general usage see page above
    ]
    if series_num >= 0:
        cmd.extend(["-n", str(series_num)])
    cmd.append(str(vol_dir)) # input directory

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        nifti_files = list(Path(out_dir).glob("*.nii.gz"))

        if (is_single_file or merge_2d or series_num >= 0):
            if len(nifti_files) != 1:
                raise RuntimeError(
                    f"Expected a single .nii.gz output due to flags "
                    f"{'-s' if is_single_file else ''} "
                    f"{'-m' if merge_2d else ''} "
                    f"{f'-n {series_num}' if series_num >= 0 else ''}, "
                    f"but found {len(nifti_files)} files."
                )
        else:
            if len(nifti_files) < 1:
                raise RuntimeError("No NIfTI (.nii.gz) files were generated.")
            
        bval_files = list(out_dir.glob("*.bval"))
        bvec_files = list(out_dir.glob("*.bvec"))
        bval_path = str(bval_files[0]) if bval_files else None
        bvec_path = str(bvec_files[0]) if bvec_files else None

        if not bval_path or not bvec_path:
            raise RuntimeError("No bvec or bval files were generated.")

        return nifti_files[0], bval_path, bvec_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"dcm2niix failed: {e.stderr}")

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

    series_num = inquirer.prompt([
        inquirer.Text("series", message="🔢 Enter series number to convert (-n <num>) [Leave blank for all]", default="")
    ])["series"]
    series_num = int(series_num) if series_num.isdigit() else -1

    merge_answer = inquirer.prompt([
        inquirer.Confirm("merge", message="🧩 Merge 2D slices into a single NIfTI (-m y)?", default=True)
    ])
    merge_2d = merge_answer["merge"]

    single_file = inquirer.prompt([
        inquirer.Confirm("single", message="📦 Force single file output (-s y)?", default=False)
    ])["single"]

    for in_dir, out_dir in zip(input_dirs, output_dirs):
        vol_dir = Path(in_dir)
        out_path = Path(out_dir)

        print(f"Converting:\n → Input: {vol_dir}\n → Output: {out_path}")
        try:
            nifti, bval, bvec = dicom_to_niix(vol_dir, out_path, merge_2d, series_num, single_file)
            print(f" Conversion succeeded: {nifti}")
        except RuntimeError as err:
            print(f"❌ Conversion failed: {err}")

def run_cli(input_path: str, output_path: str, **kwargs):
    vol_dir = Path(input_path)
    out_dir = Path(output_path)

    merge_2d = kwargs.get("merge_2d", False)
    series_num = kwargs.get("series_number", -1)
    single_file = kwargs.get("single_file", False)

    print(f" Converting:\n → Input: {vol_dir}\n → Output: {out_dir}")
    try:
        nifti, bval, bvec = dicom_to_niix(vol_dir, out_dir, merge_2d, series_num, single_file)
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
    parser.add_argument("output", nargs="?", help="Path to output directory for NIfTI files")
    parser.add_argument("-n", "--series-number", type=int, default=-1, help="Only convert this series number (-n <num>)")
    parser.add_argument("-m", "--merge-2d", action="store_true", help="Merge 2D slices (-m y)")
    parser.add_argument("-s", "--single-file", action="store_true", help="Enable single file mode (-s y)")
    parser.add_argument("-pu", "--prompt-user", action="store_true", help="Run in interactive mode")


    args = parser.parse_args()

    if args.prompt_user:
        run_interactive()
    elif args.input and args.output:
        run_cli(args.input, args.output, **vars(args))
    else:
        print("❗ You must provide input and output paths OR use --prompt-user for interactive mode.")
        parser.print_help()
        sys.exit(1)