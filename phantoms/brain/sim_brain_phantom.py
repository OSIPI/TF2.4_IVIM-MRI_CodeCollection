import os
import shutil
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

if __name__ == "__main__":

    DIFFUSIVE_REGIME = 'diffusive'
    BALLISTIC_REGIME = 'ballistic'

    folder = os.path.dirname(__file__)

    ###########################
    #  Simulation parameters  #
    regime = DIFFUSIVE_REGIME
    snr = 200
    resolution = [3,3,3]
    #                         #
    ###########################


    # Ground truth
    nii = nib.load(os.path.join(folder,'ground_truth','hrgt_icbm_2009a_nls_3t.nii.gz'))
    segmentation = np.squeeze(nii.get_fdata()[...,-1])

    with open(os.path.join(folder,'ground_truth',regime+'_groundtruth.json'), 'r') as f:
        ivim_pars = json.load(f)
    S0 = 1

    # Sequence parameters
    bval_file = os.path.join(folder,'ground_truth',regime+'.bval')
    b = np.loadtxt(bval_file)
    if regime == BALLISTIC_REGIME:
        cval_file = bval_file.replace('bval','cval')
        c = np.loadtxt(cval_file)

    # Calculate signal
    S = np.zeros(list(np.shape(segmentation))+[b.size])

    if regime == BALLISTIC_REGIME:
        Db = ivim_pars["Db"]
        for i,(D,f,vd) in enumerate(zip(ivim_pars["D"],ivim_pars["f"],ivim_pars["vd"])):
            S[segmentation==i+1,:] = S0*((1-f)*np.exp(-b*D)+f*np.exp(-b*Db-c**2*vd**2))
    else:
        for i,(D,f,Dstar) in enumerate(zip(ivim_pars["D"],ivim_pars["f"],ivim_pars["Dstar"])):
            S[segmentation==i+1,:] = S0*((1-f)*np.exp(-b*D)+f*np.exp(-b*Dstar))

    # Resample to suitable resolution
    im = zoom(S,np.append(np.diag(nii.affine)[:3]/np.array(resolution),1),order=1)
    sz = im.shape

    # Add noise
    im_noise = np.abs(im + S0/snr*(np.random.randn(sz[0],sz[1],sz[2],sz[3])+1j*np.random.randn(sz[0],sz[1],sz[2],sz[3])))

    # Save as image and sequence parameters 
    nii_out = nib.Nifti1Image(im_noise,np.eye(4))
    base_name = os.path.join(folder,'data','{}_snr{}'.format(regime,snr))
    nib.save(nii_out,base_name+'.nii.gz')
    shutil.copyfile(bval_file,base_name+'.bval')
    if regime == BALLISTIC_REGIME:
        shutil.copyfile(cval_file,base_name+'.cval')