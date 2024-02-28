import zipfile
import os
import subprocess
import time

def download_data(force=False):
    # Check if the folder exists, and create it if not
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
        with zipfile.ZipFile('OSIPI_TF24_data_phantoms.zip', 'r') as zip_ref:
            # Extract all the contents to the destination folder
            zip_ref.extractall('.')
        while not os.path.exists('Data'):
            time.sleep(10)
        while not os.path.exists('Phantoms'):
            time.sleep(10)
        while not os.path.exists('Utilities'):
            time.sleep(10)
        time.sleep(10)