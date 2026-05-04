'''
Preprocessing of brain IVIM data
Requirements: 
- ivim
- fsl
- freesurfer
- itk
- itk-elastix
'''
#%%
import numpy as np
import nibabel as nib
import subprocess, os
from nipype.interfaces import fsl
import itk
from ivim.io.base import read_bval, write_bval
from ivim.preproc.base  import extract, combine, average
from ivim.preproc import signal_drift


def denoise_wrap(im_file):
    ''' Denoising '''
    im_file_denoised = im_file.replace('.nii.gz','-pca.nii.gz')
    noise_level = im_file.replace('.nii.gz','-noise_level.nii.gz')
    print('Denoising...')
    subprocess.run(['dwidenoise', '-noise', noise_level, im_file, im_file_denoised])
    subprocess.run(['cp',im_file.replace('.nii.gz','.bval'),im_file_denoised.replace('.nii.gz','.bval')])
    subprocess.run(['cp',im_file.replace('.nii.gz','.bvec'),im_file_denoised.replace('.nii.gz','.bvec')])
    return im_file_denoised

def degibbs_wrap(im_file):
    ''' Gibbs ringing artifact removal '''
    im_file_degibbs = im_file.replace('.nii.gz','-gib.nii.gz')
    print('Degibbsing...')
    subprocess.run(['mrdegibbs', im_file, im_file_degibbs])
    subprocess.run(['cp',im_file.replace('.nii.gz','.bval'),im_file_degibbs.replace('.nii.gz','.bval')])
    subprocess.run(['cp',im_file.replace('.nii.gz','.bvec'),im_file_degibbs.replace('.nii.gz','.bvec')])
    return im_file_degibbs

def extract_volume(in_file, t_index):
    '''
    Extracts a 3D volume at time index t_index from a 4D image.
    Returns the path to the temporary file.
    '''
    out_file = in_file.replace('.nii.gz',f'-tmp{t_index}.nii.gz')
    subprocess.run(['fslroi',in_file,out_file,str(t_index),'1'])
    return out_file

def register_3D_to_ref(in_file, ref_idx, out_file,method='rigid'):
    '''
    Registers a 3D image to a reference image using ITK Elastix.
    '''
    PixelType = itk.F
    Dimension3D = 3
    Image3DType = itk.Image[PixelType, Dimension3D]

    # Built-in default parameter map seems to work well for brain
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(parameter_object.GetDefaultParameterMap(method))

    # Load reference image
    tmp_ref = extract_volume(in_file, ref_idx)
    ref_itk = itk.imread(tmp_ref, pixel_type=PixelType)

    # Loop through all volumes and register to reference
    n_vols = nib.load(in_file).shape[3]
    registered_vols = []
    for t in range(n_vols):
        if t == ref_idx:
            registered_vols.append(tmp_ref)  # No need to register reference to itself
            continue
        
        # Load moving image
        tmp_moving = extract_volume(in_file, t)
        moving_itk = itk.imread(tmp_moving, pixel_type=PixelType)

        print(f'Registering volume {t} to reference volume {ref_idx}...')
        elastix = itk.ElastixRegistrationMethod.New(ref_itk, moving_itk)
        # elastix.SetFixedImage(ref_itk)
        # elastix.SetMovingImage(moving_itk)
        elastix.SetParameterObject(parameter_object)
        elastix.Update()
        result = elastix.GetOutput()
        tmp_out = out_file.replace('.nii.gz',f'-tmp{t}.nii.gz')
        itk.imwrite(result, tmp_out)
        os.remove(tmp_moving)  # Clean up temp moving image
        registered_vols.append(tmp_out)
                    
    out_file = in_file.replace('.nii.gz','-rigid.nii.gz')
    # Stack back to 4D
    subprocess.run(['fslmerge','-t',out_file]+registered_vols)

    # Clean up
    os.remove(tmp_ref)  # Clean up temp reference image
    for t in range(1,n_vols):
        os.remove(out_file.replace('-rigid.nii.gz',f'-reg-tmp{t}.nii.gz'))  # Clean up temp registered images
    return out_file

def topup_wrap(im_file,bval_file, b0rev_file, b0rev_bval_file):
    ''' Susceptibility distortion correction '''
    print('Running topup...')
    im_file_unwarp = im_file.replace('.nii.gz','-unwarp.nii.gz')
    
    # Prepare b0:s for topup
    extract(im_file=im_file, bval_file=bval_file, outbase=im_file.replace('.nii.gz','-b0'),b_ex=0) # we dont perform averaging since it seems preferable to not do it
    combine(dwi_files=[im_file.replace('.nii.gz','-b0.nii.gz'),b0rev_file],
            bval_files=[im_file.replace('.nii.gz','-b0.bval'),b0rev_bval_file], 
            outbase=im_file.replace('.nii.gz','-b0-b0rev'))
    b = read_bval(im_file.replace('.nii.gz','-b0.bval'))
    brev = read_bval(b0rev_bval_file)
    interbase = im_file.replace('.nii.gz','-b0-b0rev')
    acqp_file = interbase + '_acqparams.txt'
    with open(acqp_file,'w') as f:
        for _ in range(np.sum(b==0)):
            f.write('0 1 0 0.050\n')
        for _ in range(np.sum(brev==0)):
            f.write('0 -1 0 0.050\n')

    topup_cmd = f'topup --imain={im_file.replace(".nii.gz","-b0-b0rev.nii.gz")} --datain={acqp_file} --config=b02b0.cnf --subsamp=1 --out={interbase} --verbose'
    os.system(topup_cmd)

    applytopup_cmd = f'applytopup --imain={im_file} --datain={acqp_file} --inindex=1 --topup={interbase} --out={im_file_unwarp} --method=jac --verbose'
    os.system(applytopup_cmd)   
    subprocess.run(['cp',im_file.replace('.nii.gz','.bval'),im_file_unwarp.replace('.nii.gz','.bval')])

    return im_file_unwarp

def brain_preproc(im_file: str, bval_file: str, b0rev_file: str, b0rev_bval_file: str, temp_folder: str):
    
    # Denoising
    im_file_denoised = denoise_wrap(im_file)
    
    # Degibbsing
    im_file_degibbs = degibbs_wrap(im_file_denoised)

    # Registration to b0
    im_file_reg = register_3D_to_ref(im_file_degibbs, ref_idx=0, out_file=im_file_degibbs.replace('.nii.gz','-reg.nii.gz'))

    # Suseptibility distortion correction 
    im_file_unwarp = topup_wrap(im_file_reg, bval_file, b0rev_file, b0rev_bval_file)

    # Brain extraction
    brain_mask = im_file_unwarp.replace('.nii.gz','-brain-mask.nii.gz')
    extract(im_file_unwarp, im_file_unwarp.replace('.nii.gz','.bval'), outbase=im_file_unwarp.replace('.nii.gz','-b0'), b_ex=0)
    average(im_file_unwarp.replace('.nii.gz','-b0.nii.gz'), im_file_unwarp.replace('.nii.gz','-b0.bval'), im_file_unwarp.replace('.nii.gz','-b0-avr'))
    subprocess.run(['mri_synthstrip', '-i', im_file_unwarp.replace('.nii.gz','-b0-avr.nii.gz'), '-o', im_file_unwarp.replace('.nii.gz', '-brain.nii.gz'), '-m', im_file_unwarp.replace('.nii.gz', '-brain-mask.nii.gz')])
    os.remove(im_file_unwarp.replace('.nii.gz','-b0.nii.gz'))
    os.remove(im_file_unwarp.replace('.nii.gz','-b0.bval'))
    os.remove(im_file_unwarp.replace('.nii.gz','-b0-avr.nii.gz'))

    # Signal drift correction
    signal_drift.spatiotemporal(im_file = im_file_unwarp, 
                                bval_file = im_file_unwarp.replace('.nii.gz','.bval'), 
                                outbase = im_file_unwarp.replace('.nii.gz','-sigdri'), 
                                roi_file = brain_mask)
    subprocess.run(['cp',im_file_unwarp.replace('.nii.gz','.bval'),im_file_unwarp.replace('.nii.gz','-sigdri_corr.bval')])
#%%
# im_file = os.getcwd()+'/temp/diffusive_snr200.nii.gz'
# bval_file = os.getcwd()+'/temp/diffusive_snr200.bval'
# b0rev_file = os.getcwd()+'/temp/diffusive_snr200_b0.nii.gz'
# b0rev_bval_file = os.getcwd()+'/temp/diffusive_snr200_b0.bval'

# im_file_denoised = denoise_wrap(im_file)
# im_file_degibbs = degibbs_wrap(im_file_denoised)
# im_file_reg = register_3D_to_ref(im_file_degibbs, ref_idx=0, out_file=im_file_degibbs.replace('.nii.gz','-reg.nii.gz'))
# im_file_unwarp = topup_wrap(im_file_reg, bval_file, b0rev_file, b0rev_bval_file)

# Brain extraction
# brain_mask = im_file_unwarp.replace('.nii.gz','-brain-mask.nii.gz')
# extract(im_file_unwarp, bval_file, outbase=im_file_unwarp.replace('.nii.gz','-b0'), b_ex=0)
# average(im_file_unwarp.replace('.nii.gz','-b0.nii.gz'), im_file_unwarp.replace('.nii.gz','-b0.bval'), im_file_unwarp.replace('.nii.gz','-b0-avr'))
# subprocess.run(['mri_synthstrip', '-i', im_file_unwarp.replace('.nii.gz','-b0-avr.nii.gz'), '-o', im_file_unwarp.replace('.nii.gz', '-brain.nii.gz'), '-m', im_file_unwarp.replace('.nii.gz', '-brain-mask.nii.gz')])
# os.remove(im_file_unwarp.replace('.nii.gz','-b0.nii.gz'))
# os.remove(im_file_unwarp.replace('.nii.gz','-b0.bval'))
# os.remove(im_file_unwarp.replace('.nii.gz','-b0-avr.nii.gz'))

# # Signal drift correction
# signal_drift.spatiotemporal(im_file = im_file_unwarp, 
#                             bval_file = bval_file, 
#                             outbase = im_file_unwarp.replace('.nii.gz','-sigdri'), 
#                             roi_file = brain_mask)
# subprocess.run(['cp',im_file_unwarp.replace('.nii.gz','.bval'),im_file_unwarp.replace('.nii.gz','-sigdri_corr.bval')])

# %%
