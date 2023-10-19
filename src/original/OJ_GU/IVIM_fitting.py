import os
import matlab.engine

def IVIM_seg(Y, b, lim, blim, disp_prog):
    eng = matlab.engine.start_matlab()
    s = eng.genpath(os.path.dirname(__file__))
    eng.addpath(s, nargout=0)
    pars = eng.IVIM_seg(Y, b, lim, blim, disp_prog)
    eng.quit()
    return pars

def IVIM_bayes(Y, f, D, Dstar, S0, b, lim, n, rician, prior, burns, meanonly):
    eng = matlab.engine.start_matlab()
    s = eng.genpath(os.path.dirname(__file__))
    eng.addpath(s, nargout=0)
    out = eng.IVIM_bayes(Y, f, D, Dstar, S0, b, lim, n, rician, prior, burns, meanonly)
    eng.quit()
    return out