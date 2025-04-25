from pathlib import Path
import numpy as np
import nibabel as nib
from utilities.data_simulation.GenerateData import GenerateData
from WrapImage.nifti_wrapper import save_nifti_file
from WrapImage.dicom2niix_wrapper import save_dicom_objects

def save_bval_bvec(filename, values):
    if filename.endswith('.bval'):
        # Convert list to a space-separated string for bval
        values_string = ' '.join(map(str, values))
    elif filename.endswith('.bvec'):
        # Convert 2D list to a line-separated, space-separated string for bvec
        values_string = '\n'.join(' '.join(map(str, row)) for row in values)
    else:
        raise ValueError("Unsupported file extension. Use '.bval' or '.bvec'.")

    with open(filename, 'w') as file:
        file.write(values_string)

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
signals = gd.ivim_signal(D_in, Dp_in, f_in, S0, bvals_reshaped)


# Generate 4D signal by stacking volumes for each bval
signals_4d = np.stack([
    gd.ivim_signal(D_in, Dp_in, f_in, S0, np.full(shape, bval))
    for bval in bvals
], axis=-1)

# Save the generated image as a NIfTI file
save_nifti_file(signals, "ivim_simulation.nii.gz")
# Save the bval in a file
save_bval_bvec("ivim_simulation.bval", [0, 50, 100, 500, 1000])
# Save the bvec value 
save_bval_bvec("ivim_simulation.bvec", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

bvecs = [[1, 0, 0]] * len(bvals)  # Assuming all x-direction

save_dicom_objects(
    volume_4d=signals_4d,
    bvals=bvals.tolist(),
    bvecs=bvecs,
    out_dir=Path("ivim_simulation_dicom"),
    f_vals=f_in,
    D_vals=D_in,
    Dp_vals=Dp_in
)