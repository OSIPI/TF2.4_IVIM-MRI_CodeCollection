'''
1. Load brain phantom
2. Add Rician noise
3. Denoise using denoise algorithm
4. Check that sum of squared residuals decreased after denoising
'''
import os

import numpy as np
from phantoms.brain.sim_brain_phantom import simulate_brain_phantom
import nibabel as nib
import matplotlib.pyplot as plt
from src.original.preprocessing.EP_GU.brain_pipeline import denoise_wrap


def test_denoise_wrap():
    # Simulate phantom with noise
    snr = 50
    simulate_brain_phantom(snr = snr)
    
    # Denoising
    im_file = 'phantoms/brain/data/diffusive_snr{}.nii.gz'.format(snr)
    im_file_denoised = denoise_wrap(im_file)
    S_denoised = nib.load(im_file_denoised).get_fdata()

    # Compute sum of squared residuals before denoising
    mask = nib.load(os.path.join('phantoms','brain','data','diffusive_snr{}_mask.nii.gz'.format(snr))).get_fdata()
    ref_file = 'phantoms/brain/data/diffusive_reference.nii.gz'
    S = nib.load(im_file).get_fdata()
    S_ref = nib.load(ref_file).get_fdata()
    ssr_before = np.sum((S[mask!=0,:]-S_ref[mask!=0,:])**2)
    ssr_after = np.sum((S_denoised[mask!=0,:]-S_ref[mask!=0,:])**2)

    

    # print('SSR before denoising: {}'.format(ssr_before))
    # print('SSR after denoising: {}'.format(ssr_after))

    # plt.subplot(1,3,1)
    # plt.imshow(np.rot90(S_ref[:,:,30,10]),cmap='gray')
    # plt.title('Reference')
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(1,3,2)
    # plt.imshow(np.rot90(S[:,:,30,10]),cmap='gray')
    # plt.title('Noisy')
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(1,3,3)
    # plt.imshow(np.rot90(S_denoised[:,:,30,10]),cmap='gray')
    # plt.title('Denoised')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    # Check that sum of squared residuals decreased after denoising
    if ssr_after < ssr_before:
        assert True
    else:   
        assert False



