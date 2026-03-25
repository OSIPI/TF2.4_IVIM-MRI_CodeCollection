'''
Not done! Need a better metric than just noise correlation

'''
# %%
from phantoms.brain.sim_brain_phantom_preproc import simulate_brain_phantom

simulate_brain_phantom(snr = 50)
#%%
import nibabel as nib
import numpy as np
from src.original.EP_GU.brain_pipeline import degibbs_wrap
import matplotlib.pyplot as plt
import os

# load brain phantom
S_ref = nib.load('phantoms/brain/data/diffusive_reference_relax.nii.gz').get_fdata()

# Convert to k-space 
S_k = np.fft.fftn(S_ref,axes=(0,1))
S_k = np.fft.fftshift(S_k,axes=(0,1))

plt.subplot(1,2,1)
plt.imshow(np.log(np.abs(S_k[:,:,30,0])+1),cmap='gray')
plt.xticks([])
plt.yticks([])

# Crop k-space
S_k[:10,:,:,:] = 0
S_k[-10:,:,:,:] = 0
# S_k[:,:10,:,:] = 0
# S_k[:,-10:,:,:] = 0

# Test plot
plt.subplot(1,2,2)
plt.imshow(np.log(np.abs(S_k[:,:,30,0])+1),cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

# Inverse Fourier transform
S_crop = np.fft.ifftshift(S_k,axes=(0,1))
S_crop = np.fft.ifftn(S_crop,axes=(0,1)).real

# Save image
S_crop_nii = nib.Nifti1Image(S_crop.astype(np.float32),np.eye(4))
S_crop_file = 'phantoms/brain/data/diffusive_reference_relax_cropped.nii.gz'
nib.save(S_crop_nii,S_crop_file)

# Apply degibbs
S_degibbs_file = degibbs_wrap(S_crop_file)

# Plot results
S_degibbs = nib.load(S_degibbs_file).get_fdata()
plt.subplot(1,3,1)
plt.imshow(np.rot90(S_ref[:,:,30,0]),cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,2)
plt.imshow(np.rot90(S_crop[:,:,30,0]),cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,3)
plt.imshow(np.rot90(S_degibbs[:,:,30,0]),cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

mask = nib.load(os.path.join('phantoms','brain','data','diffusive_snr50_relax_mask.nii.gz')).get_fdata()

def noise_correlation(img):
    img = img - np.mean(img)
    return np.sum(img[:, :-1] * img[:, 1:]) / np.sum(img**2)

S_crop = nib.load(S_crop_file).get_fdata()
S_degibbs = nib.load(S_degibbs_file).get_fdata()
S_crop[mask==0,:]=0
S_degibbs[mask==0,:]=0

print('Noise correlation before degibbs: {}'.format(noise_correlation(S_crop)))
print('Noise correlation after degibbs: {}'.format(noise_correlation(S_degibbs)))