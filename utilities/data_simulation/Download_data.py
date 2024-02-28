import zipfile
import os
import subprocess
import time


def unzip_file(zip_file_path, extracted_folder_path):
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract each file one by one
        for file_info in zip_ref.infolist():
            zip_ref.extract(file_info, extracted_folder_path)


def download_data(force=False):
    # Check if the folder exists, and create it if not
    curdir=os.getcwd()
    base_folder = os.path.abspath(os.path.dirname(__file__))
    base_folder = os.path.split(os.path.split(base_folder)[0])[0]
    if not os.path.exists(os.path.join(base_folder,'download')):
        os.makedirs(os.path.join(base_folder,'download'))
        print(f"Folder '{'download'}' created.")
        # Change to the specified folder
        os.chdir(os.path.join(base_folder,'download'))
    subprocess.run(["zenodo_get", 'https://zenodo.org/records/10696605'])
    while not os.path.exists('OSIPI_TF24_data_phantoms.zip'):
        time.sleep(1)
    # Open the zip file
    if force or not os.path.exists('Data'):
        # Unzip the file
        unzip_file('OSIPI_TF24_data_phantoms.zip', '.')
        # Wait for the extraction to complete by checking for the existence of any file
        # Wait for the extraction to complete by checking for expected subdirectories
        expected_subdirectories = [os.path.join('Utilities','DRO.npy'),os.path.join('Data','abdomen.nii.gz'), os.path.join('Data','brain.nii.gz'), os.path.join('Phantoms','brain','data','ballistic_snr200.nii.gz'), os.path.join('Phantoms','brain','data','diffusive_snr200.nii.gz'), os.path.join('Phantoms','XCAT_MAT_RESP','XCAT5D_RP_1_CP_1.mat'), os.path.join('Phantoms','XCAT_MAT_RESP','XCAT5D_RP_9_CP_1.mat'), os.path.join('Phantoms','XCAT_MAT_RESP','XCAT5D_RP_20_CP_1.mat')]  # Add the expected subdirectories
        while not all(
                os.path.isfile(subdir) for subdir in expected_subdirectories):
            time.sleep(1)  # Wait for 1 second
        time.sleep(10)
    os.chdir(curdir)