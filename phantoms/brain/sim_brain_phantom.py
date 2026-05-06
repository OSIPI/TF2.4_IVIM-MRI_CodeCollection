import os
import shutil
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from utilities.data_simulation.Download_data import download_data

DIFFUSIVE_REGIME = 'diffusive'
BALLISTIC_REGIME = 'ballistic'

def simulate_brain_phantom(regime=DIFFUSIVE_REGIME, snr=100, sim_relaxation=True, TE=60e-3, TR=5, resolution=[3,3,3]):
    '''
    Simulation parameters can be set by changing the default values of the function arguments. 
    The default values are chosen to be suitable for a diffusive regime phantom, but can be changed to simulate a ballistic regime phantom as well. 
    The simulated image is saved in phantoms/brain/data with the name 'diffusive_sn{snr}.nii.gz' or 'ballistic_sn{snr}.nii.gz' depending on the regime. 
    The corresponding bvals and cvals (if applicable) are also saved in the same folder.
    
    regime: 'diffusive' or 'ballistic' to choose the type of phantom to simulate
    snr: signal-to-noise ratio of the simulated image
    sim_relaxation: whether to simulate T1 and T2 relaxation effects in the signal
    TE: echo time in seconds
    TR: repetition time in seconds
    resolution: voxel size in mm
    '''
    download_data()


    folder = os.path.dirname(__file__)

    # Ground truth
    nii = nib.load(os.path.join(os.path.split(os.path.split(folder)[0])[0],'download','Phantoms','brain','ground_truth','hrgt_icbm_2009a_nls_3t.nii.gz'))
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
    print(ivim_pars)
    if regime == BALLISTIC_REGIME:
        Db = ivim_pars["Db"]
        for i,(D,f,vd,T1,T2) in enumerate(zip(ivim_pars["D"],ivim_pars["f"],ivim_pars["vd"],ivim_pars["T1"],ivim_pars["T2"])):
            S[segmentation==i+1,:] = S0*((1-f)*np.exp(-b*D)+f*np.exp(-b*Db-c**2*vd**2))
            if sim_relaxation:
                S[segmentation==i+1,:] *= np.exp(-TE/T2)*(1-np.exp(-TR/T1))
    else:
        for i,(D,f,Dstar,T1,T2) in enumerate(zip(ivim_pars["D"],ivim_pars["f"],ivim_pars["Dstar"],ivim_pars["T1"],ivim_pars["T2"])):
            S[segmentation==i+1,:] = S0*((1-f)*np.exp(-b*D)+f*np.exp(-b*Dstar))
            if sim_relaxation:
                S[segmentation==i+1,:] *= np.exp(-TE/T2)*(1-np.exp(-TR/T1))

    # Resample to suitable resolution
    im = zoom(S,np.append(np.diag(nii.affine)[:3]/np.array(resolution),1),order=1)
    sz = im.shape

    # Save image without noise for reference
    nii_out = nib.Nifti1Image(im,np.eye(4))
    base_name = os.path.join(folder,'data','{}_reference'.format(regime,snr))
    nib.save(nii_out,base_name+'.nii.gz')
    shutil.copyfile(bval_file,base_name+'.bval')

    # Add Rician noise
    im_noise = np.abs(im + S0/snr*(np.random.randn(sz[0],sz[1],sz[2],sz[3])+1j*np.random.randn(sz[0],sz[1],sz[2],sz[3])))

    # Save as image and sequence parameters 
    nii_out = nib.Nifti1Image(im_noise,np.eye(4))
    base_name = os.path.join(folder,'data','{}_snr{}'.format(regime,snr))
    nib.save(nii_out,base_name+'.nii.gz')
    shutil.copyfile(bval_file,base_name+'.bval')
    if regime == BALLISTIC_REGIME:
        shutil.copyfile(cval_file,base_name+'.cval')

    # Resample and save segmentation
    segmentation = zoom(segmentation,np.diag(nii.affine)[:3]/np.array(resolution),order=0)
    nii_out = nib.Nifti1Image(segmentation,np.eye(4))
    base_name = os.path.join(folder,'data','{}_snr{}_mask'.format(regime,snr))
    nib.save(nii_out,base_name+'.nii.gz')


if __name__ == "__main__":
    simulate_brain_phantom()
