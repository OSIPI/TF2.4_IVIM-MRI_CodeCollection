import numpy as np
import nibabel as nib
from utilities.data_simulation.GenerateData import GenerateData

def save_nii(data, filename='ivim_image.nii.gz'):
    """
    Save the data as a NIfTI file (.nii.gz)
    """
    nii_img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nii_img, filename)

# Set random seed for reproducibility
np.random.seed(42)

# Create GenerateData instance
gd = GenerateData()

# Generate random input data
shape = (10, 10, 5)
f_in = np.random.uniform(low=0, high=1, size=shape)
D_in = np.random.uniform(low=0, high=1e-3, size=shape)
Dp_in = np.random.uniform(low=0, high=1e-1, size=shape)
S0 = 1000  # Setting a constant S0 for simplicity
bvals = np.array([0, 50, 100, 500, 1000])
bvals_reshaped = np.broadcast_to(bvals, shape)

# Generate IVIM signal
images = gd.ivim_signal(D_in, Dp_in, f_in, S0, bvals_reshaped)

# Save the generated image as a NIfTI file
save_nii(images)
