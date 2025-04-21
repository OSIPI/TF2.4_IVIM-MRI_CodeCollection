import os
from pathlib import Path
import subprocess
import yaml
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


def generate_batch_config(
    input_dirs,
    output_dirs,
    config_path="batch_config.yaml"
):
    """
    input_dirs: List of input directory paths
    output_dirs: Single output directory (string) OR list of output directories
    """

    # Normalize inputs
    if isinstance(output_dirs, str):
        output_dirs = [output_dirs] * len(input_dirs)
    elif isinstance(output_dirs, list):
        if len(input_dirs) != 1 and len(output_dirs) != len(input_dirs):
            raise ValueError("Number of output directories must match number of input directories, unless only one output directory is provided.")

    config = {
        "Options": {
            "isGz": True, # compressed nii.gz
            "isFlipY": False, # flip Y-axis of images
            "isVerbose": False, # by default is verbose; this value does not change anything
            "isCreateBIDS": True, # create BIDS-compatible NIfTI and JSON files
            "isOnlySingleFile": False
        },
        "Files": []
    }

    for in_dir, out_dir in zip(input_dirs, output_dirs):
        if not os.path.isdir(in_dir):
            print(f"Warning: {in_dir} is not a valid directory.")
            continue

        files = os.listdir(in_dir)
        if not files:
            print(f"No files found in {in_dir}")
            continue

        files = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
        if not files:
            print(f"No files found in {in_dir}")
            continue

        for f in files:
            filename = os.path.splitext(f)[0].replace(" ", "_").lower()

            config["Files"].append({
                "in_dir": os.path.abspath(in_dir),
                "out_dir": os.path.abspath(out_dir),
                "filename": filename
            })

    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    print(f"Config written to {config_path}")

def dicom_to_niix(vol_dir: Path, out_dir: Path, merge_2d: bool = False):
    """
    For converting DICOM images to a (compresssed) 4d nifti image
    """
    os.makedirs(out_dir, exist_ok=True)

    try:
        res = subprocess.run(
            [
                "dcm2niix",
                "-f", "%s_%p", # dcm2niix attempts to provide a sensible file naming scheme
                "-o", out_dir, # output directory
                "-z", "y", #specifying compressed nii.gz file
                "-m", "y" if merge_2d else "n",  # Add merge option
                # https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage
                # for further configuration for general usage see page above
                vol_dir # input directory
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        nifti_files = list(Path(out_dir).glob("*.nii.gz"))
        if not nifti_files:
            raise RuntimeError("No NIfTI (.nii.gz) files were generated.")

        bval_files = list(out_dir.glob("*.bval"))
        bvec_files = list(out_dir.glob("*.bvec"))
        bval_path = str(bval_files[0]) if bval_files else None
        bvec_path = str(bvec_files[0]) if bvec_files else None


        return nifti_files[0], bval_path, bvec_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"dcm2niix failed: {e.stderr}")

if __name__ == "__main__":
    input_dirs = []
    output_dirs = []

    print("Inquiring to convert your DICOM Images to NIfTI Image(s) \n")

    while True:
        input_dir = prompt_input_directory()
        output_dir = prompt_output_directory(input_dir)

        input_dirs.append(input_dir)
        output_dirs.append(output_dir)

        add_more = inquirer.prompt([
            inquirer.Confirm(
                "more",
                message="➕ Do you want to add another input/output pair?",
                default=False
            )
        ])["more"]

        if not add_more:
            break

    if len(input_dirs) == 1:
        print("📥 Single input/output pair detected — using `dcm2niix`...")
        vol_dir = Path(input_dirs[0])
        out_dir = Path(output_dirs[0])

        merge_answer = inquirer.prompt([
            inquirer.Confirm(
                "merge",
                message="🧩 Do you want to merge 2D slices into a single NIfTI (-m y)?",
                default=True
            )
        ])
        merge_2d = merge_answer["merge"]

        try:
            nifti, bval, bvec = dicom_to_niix(vol_dir, out_dir, merge_2d=merge_2d)
            print(f"\n✅ NIfTI file created: {nifti}")
            if bval:
                print(f"  📈 bval: {bval}")
            if bvec:
                print(f"  📊 bvec: {bvec}")

            post_process = inquirer.prompt([
                inquirer.Confirm(
                    "run_post",
                    message="🧩 Do you want to run IVIM fit algorithm on the NIfTI file now?",
                    default=True
                )
            ])["run_post"]

            if bvec and bval and post_process:
                print("\n🧩 Running post-processing: OSIPI IVIM fitting...\n")
                subprocess.run([
                    "python3", "-m", "WrapImage.nifti_wrapper",
                    str(nifti),
                    str(bvec),
                    str(bval),
                ], check=True)
                print("✅ NIfTI post-processing completed.")
        except RuntimeError as err:
            print(f"❌ Conversion failed: {err}")

    else:
        print("📥📥📥 Multiple inputs detected — generating batch config and using `dcm2niibatch`...")
        config_path = "batch_config.yaml"
        generate_batch_config(input_dirs, output_dirs, config_path=config_path)

        try:
            with subprocess.Popen(
                ["dcm2niibatch", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            ) as proc:
                # for line in proc.stdout: # uncomment this to see verbose logs
                #     print(line.strip())
                proc.wait()
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, proc.args)

                print("✅ Batch conversion completed successfully.")

        except subprocess.CalledProcessError as err:
            print(f"❌ Batch conversion failed:\n{err}")

