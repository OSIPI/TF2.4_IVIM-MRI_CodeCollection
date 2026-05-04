"""
September 2020 by Oliver Gurney-Champion & Misha Kaandorp
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved unsupervised physics-informed deep learning for intravoxel-incoherent motion modeling and evaluation in pancreatic cancer patients. MRM 2021)

requirements:
numpy
torch
tqdm
matplotlib
scipy
joblib
"""

# this loads all patient data and evaluates it all.
import os
import time
import nibabel as nib
import numpy as np
import IVIMNET.deep as deep
import torch
from IVIMNET.fitting_algorithms import fit_dats
from hyperparams import hyperparams as hp

arg = hp()
arg = deep.checkarg(arg)

testdata = False

### folder patient data
folder = 'example_data'

### load patient data
print('Load patient data \n')
# load and init b-values
text_file = np.genfromtxt('{folder}/bvalues.bval'.format(folder=folder))
bvalues = np.array(text_file)
selsb = np.array(bvalues) == 0
# load nifti
data = nib.load('{folder}/data.nii.gz'.format(folder=folder))
datas = data.get_fdata() 
# reshape image for fitting
sx, sy, sz, n_b_values = datas.shape 
X_dw = np.reshape(datas, (sx * sy * sz, n_b_values))

### select only relevant values, delete background and noise, and normalise data
S0 = np.nanmean(X_dw[:, selsb], axis=1)
S0[S0 != S0] = 0
S0 = np.squeeze(S0)
valid_id = (S0 > (0.5 * np.median(S0[S0 > 0]))) 
datatot = X_dw[valid_id, :]
# normalise data
S0 = np.nanmean(datatot[:, selsb], axis=1).astype('<f')
datatot = datatot / S0[:, None]
print('Patient data loaded\n')

### least square fitting
if arg.fit.do_fit:
    print('Conventional fitting\n')
    start_time = time.time()
    paramslsq = fit_dats(bvalues.copy(), datatot.copy()[:, :len(bvalues)], arg.fit)
    elapsed_time1 = time.time() - start_time
    print('\ntime elapsed for lsqfit: {}\n'.format(elapsed_time1))
    # define names IVIM params
    names_lsq = ['D_{}_{}'.format(arg.fit.method, 'fitS0' if not arg.fit.fitS0 else 'freeS0'),
                 'f_{}_{}'.format(arg.fit.method, 'fitS0' if not arg.fit.fitS0 else 'freeS0'),
                 'Dp_{}_{}'.format(arg.fit.method, 'fitS0' if not arg.fit.fitS0 else 'freeS0')]
    
    tot = 0
    # fill image array
    for k in range(len(names_lsq)):
        img = np.zeros([sx * sy * sz])
        img[valid_id] = paramslsq[k][tot:(tot + sum(valid_id))]
        img = np.reshape(img, [sx, sy, sz])
        nib.save(nib.Nifti1Image(img, data.affine, data.header),'{folder}/{name}.nii.gz'.format(folder=folder,name=names_lsq[k]))
    print('Conventional fitting done\n')

### NN network
if not arg.train_pars.skip_net:
    print('NN fitting\n')
    res = [i for i, val in enumerate(datatot != datatot) if not val.any()] # Remove NaN data
    start_time = time.time()
    # train network
    net = deep.learn_IVIM(datatot[res], bvalues, arg)
    elapsed_time1net = time.time() - start_time
    print('\ntime elapsed for Net: {}\n'.format(elapsed_time1net))
    start_time = time.time()
    # predict parameters
    paramsNN = deep.predict_IVIM(datatot, bvalues, net, arg)
    elapsed_time1netinf = time.time() - start_time
    print('\ntime elapsed for Net inf: {}\n'.format(elapsed_time1netinf))
    print('\ndata length: {}\n'.format(len(datatot)))
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    # define names IVIM params
    names = ['D_NN_{nn}_lr_{lr}'.format(nn=arg.save_name, lr=arg.train_pars.lr),
             'f_NN_{nn}_lr_{lr}'.format(nn=arg.save_name, lr=arg.train_pars.lr),
             'Dp_NN_{nn}_lr_{lr}'.format(nn=arg.save_name, lr=arg.train_pars.lr),]
    tot = 0
    # fill image array and make nifti
    for k in range(len(names)):
        img = np.zeros([sx * sy * sz])
        img[valid_id] = paramsNN[k][tot:(tot + sum(valid_id))]
        img = np.reshape(img, [sx, sy, sz])
        nib.save(nib.Nifti1Image(img, data.affine, data.header),'{folder}/{name}.nii.gz'.format(folder = folder,name=names[k])),
    print('NN fitting done\n')
