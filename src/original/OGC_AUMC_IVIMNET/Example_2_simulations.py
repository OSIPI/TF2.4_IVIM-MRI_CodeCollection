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

# import
import numpy as np
import IVIMNET.simulations as sim
import IVIMNET.deep as deep
from hyperparams import hyperparams as hp_example

# load hyperparameter
arg = hp_example()
arg = deep.checkarg(arg)

matlsq = np.zeros([len(arg.sim.SNR), 3, 3])
matNN = np.zeros([len(arg.sim.SNR), 3, 3])
stability = np.zeros([len(arg.sim.SNR), 3])
a = 0

for SNR in arg.sim.SNR:
    print('\n simulation at SNR of {snr}\n'.format(snr=SNR))
    if arg.fit.do_fit:
        matlsq[a, :, :], matNN[a, :, :], stability[a, :] = sim.sim(SNR, arg)
        print('\nresults from lsq:')
        print(matlsq)
    else:
        matNN[a, :, :], stability[a, :] = sim.sim(SNR, arg)
    a = a + 1
    print('\nresults from NN: columns show themean, the RMSE/mean and the Spearman coef [DvDp,Dvf,fvDp] \n'
          'the rows show D, f and D*\n'
          'and the different matixes repressent the different SNR levels {}:'.format(arg.sim.SNR))
    print(matNN)
    # if repeat is higher than 1, then print stability (CVNET)
    if arg.sim.repeats > 1:
        print('\nstability of NN for D, f and D* at different SNR levels:')
        print(stability)
