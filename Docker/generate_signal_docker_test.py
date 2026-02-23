import datetime
import os
from pathlib import Path
import sys
import uuid
import numpy as np
import nibabel as nib
import itk
from utilities.data_simulation.GenerateData import GenerateData
from WrapImage.nifti_wrapper import save_nifti_file

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

# Save the generated image as a NIfTI file
save_nifti_file(signals, "ivim_simulation.nii.gz")
# Save the bval in a file
save_bval_bvec("ivim_simulation.bval", [0, 50, 100, 500, 1000])
# Save the bvec value 
save_bval_bvec("ivim_simulation.bvec", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def save_dicom_files():
    os.makedirs("ivim_simulation", exist_ok=True)
    InputImageType = itk.Image[itk.D, 3] 
    ReaderType = itk.ImageFileReader[InputImageType]
    NiftiImageIOType = itk.NiftiImageIO.New()


    reader = ReaderType.New()
    reader.SetImageIO(NiftiImageIOType)
    reader.SetFileName("ivim_simulation.nii.gz")

    try:
        reader.Update()
    except Exception as e:
        print(f"Error occured while reading NIfTI in ivim_simulation: {e}")
        sys.exit(1)

    OutputPixelType = itk.SS
    # The casting filter output image type will be 3D with the new pixel type
    CastedImageType = itk.Image[OutputPixelType, 3]
    CastFilterType = itk.CastImageFilter[InputImageType, CastedImageType]
    caster = CastFilterType.New()
    caster.SetInput(reader.GetOutput())
    caster.Update()


    OutputImageType = itk.Image[OutputPixelType, 2]
    FileWriterType = itk.ImageSeriesWriter[CastedImageType, OutputImageType]
    GDCMImageIOType = itk.GDCMImageIO.New()
    writer = FileWriterType.New()
    size = reader.GetOutput().GetLargestPossibleRegion().GetSize()
    fnames = itk.NumericSeriesFileNames.New()
    num_slices = size[2]

    fnames.SetStartIndex(0)
    fnames.SetEndIndex(num_slices - 1)  # Iterate over the Z dimension (slices)
    fnames.SetIncrementIndex(1)
    fnames.SetSeriesFormat(os.path.join("ivim_simulation", f"ivim_simulation_%04d.dcm"))

    # meta_dict = itk.MetaDataDictionary()
    # include correct headers here to be tuned for Vendor
    # GDCMImageIOType.SetMetaDataDictionary(meta_dict)
    # GDCMImageIOType.KeepOriginalUIDOn()
    writer.SetInput(caster.GetOutput())
    writer.SetImageIO(GDCMImageIOType)
    writer.SetFileNames(fnames.GetFileNames())
    try:
        writer.Write()
    except Exception as e:
        print(f"Error occurred while writing DICOMs in ivim simulation: {e}")
        sys.exit(1)

args = sys.argv[1:]
if "--dicom" in args:
    # read the generated nii file to dicom files
    save_dicom_files()
    # Save the bval in a file
    save_bval_bvec(os.path.join("ivim_simulation","ivim_simulation.bval"), [0, 50, 100, 500, 1000])
    # Save the bvec value 
    save_bval_bvec(os.path.join("ivim_simulation","ivim_simulation.bvec"), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
