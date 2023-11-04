import os
import numpy as np

# Data from Ahlgren et al 2016 NMRinBiomed
Delta = 7.5e-3 # 7.5 ms
delta = 7.3e-3 # 7.3 ms

# For DDE sequence we have
# b = 2y^2G^2d^2(D-d/3), c = 2yGdD => c = sqrt(b 2D^2/(D-d/3))
bval_file = os.path.join(os.path.dirname(__file__),'ballistic.bval')
b = np.loadtxt(bval_file)
c = np.sqrt(b*2*Delta**2/(Delta-delta/3))
c[1:(c.size-1)//2+1] = 0 # flow compensated => c = 0
cval_file = bval_file.replace('bval','cval')
np.savetxt(cval_file,c,fmt='%.3f',newline=' ')
