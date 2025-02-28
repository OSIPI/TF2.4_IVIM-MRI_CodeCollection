import zipfile
import os
import subprocess
import time
import zenodo_get

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
    subprocess.check_call(["zenodo_get", 'https://zenodo.org/records/14605039'])
    # Open the zip file
    if force or not os.path.exists('Data'):
        # Unzip the file
        unzip_file('OSIPI_TF24_data_phantoms.zip', '.')
    os.chdir(curdir)

if __name__ == "__main__":
    download_data(force=True)