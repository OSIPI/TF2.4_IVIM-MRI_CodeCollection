""" Methods for outlier removal.
Reproduced with permission from Oscar Jalngefjord: https://github.com/oscarjalnefjord/ivim/tree/outlier by Elina Petersson
"""

import numpy as np
import nibabel as nib
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

NO_REGIME = 'no'
DIFFUSIVE_REGIME = 'diffusive'
BALLISTIC_REGIME = 'ballistic'
Db = 1.75e-3  # mm2/s, see Ahlgren et al 2016 NMR in Biomedicine

def roi_based(im_file: str, bval_file: str, roi_file: str, outbase: str, regime:str , fig: bool = False, cval_file: str | None = None):
    """
    Identify outliers by fit to ROI average.

    Arguments:
        im_file:    path to nifti image file
        bval_file:  path to .bval file
        roi_file:   path to nifti file defining a region-of-interest (ROI) in which the correction is calculated and applied
        outbase:    basis for output filenames, i.e. filename without file extension to which .nii.gz, .bval, etc. is added
        regime:     IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        fig:        (optional) if True, a diagnostic figure is output
        cval_file:  (optional) path to .cval file
    """

    check_regime(regime)
    Y = nib.load(im_file).get_fdata()
    if roi_file is not None:
        roi = nib.load(roi_file).get_fdata().astype(bool)
    else:
        roi = np.full(Y.shape[:-1], True)
    Y = Y[roi,:]
    b = np.atleast_1d(np.loadtxt(bval_file))
    if regime == BALLISTIC_REGIME:
        c = np.atleast_1d(np.loadtxt(cval_file))

    y_avg = np.median(Y, axis=0)

    if regime == NO_REGIME:
        def model(x, idx):
            return sIVIM(b[idx], x[0], x[1], x[2])
        x0 = [1e-3, 0.1, np.max(y_avg)]
        bounds = ((0, 3e-3), (0, 1), (0, 2*np.max(y_avg)))
    elif regime == BALLISTIC_REGIME:
        def model(x, idx):
            return ballistic(b[idx], c[idx], x[0], x[1], x[2], x[3])
        x0 = [1e-3, 0.1, 2, np.max(y_avg)]
        bounds = ((0, 3e-3), (0, 1), (0, 5), (0, 2*np.max(y_avg)))
    else: # diffusive
        def model(x, idx):
            return diffusive(b[idx], x[0], x[1], x[2], x[3])
        x0 = [1e-3, 0.1, 10e-3, np.max(y_avg)]
        bounds = ((0, 3e-3), (0, 1), (0, 1), (0, 2*np.max(y_avg)))
    

    has_outliers = True
    idx = np.full_like(b, True, dtype=bool)
    while has_outliers:
        def fun(x):
            return np.sum(np.abs(model(x, idx)-y_avg[idx]))
        
        x = minimize(fun, x0, bounds=bounds).x
        res = np.squeeze(model(x, idx)) - y_avg[idx]
        iqr = (np.quantile(res, 0.75) - np.quantile(res, 0.25))
        tmp = np.full_like(b, False, dtype=bool)
        tmp[idx] = np.abs(res) < 3*iqr
        if np.all(tmp == idx):
            has_outliers = False
        else:
            idx = tmp
    
    if fig:
        fig, ax = plt.subplots(1, 1)
        b_plot = np.linspace(np.min(b), np.max(b))
        if regime == NO_REGIME:
            y = np.squeeze(sIVIM(b_plot, x[0], x[1], x[2]))
            
        elif regime == BALLISTIC_REGIME:
            c_plot = b_plot * (c/b)[c>0][np.argmax(b[c>0])]
            y = np.squeeze(ballistic(b_plot, c_plot, x[0], x[1], x[2], x[3]))
            if np.any((b>0)&(c==0)):
                ax.plot(b_plot, np.squeeze(ballistic(b_plot, np.zeros_like(b_plot), x[0], x[1], x[2], x[3])))
        else:
            y = np.squeeze(diffusive(b_plot, x[0], x[1], x[2], x[3]))
        ax.plot(b_plot, y)
        ax.plot(b, y_avg, 'ko')
        ax.plot(b[~idx], y_avg[~idx], 'rx')
        ax.set_xlabel(r'b [s/mm$^2$]')
        ax.set_ylabel('Signal [a.u]')
        fig.savefig(outbase+'.png')
        plt.close(fig)

    sz = roi.shape

    if Y[:,idx].ndim > 1:
        im_new = np.full(list(sz) + [Y[:,idx].shape[1]], np.nan)
        im_new[roi, :] = Y[:,idx]
    else:
        im_new = np.full(sz, np.nan)
        im_new[roi] = Y[:,idx]
    nib.save(nib.Nifti1Image(im_new, nib.load(im_file).affine, nib.load(im_file).header), outbase+'.nii.gz')
    np.savetxt(outbase+'.bval', b[idx], fmt='%.1f', newline=' ')
    if regime == BALLISTIC_REGIME:
        np.savetxt(outbase+'.cval', c[idx], fmt='%.1f', newline=' ')



def at_least_1d(pars: list) -> list:
    """ Check that each parameter is atleast one dimension in shape. """
    for i, par in enumerate(pars):
        pars[i] = np.atleast_1d(par)
    return pars

def check_regime(regime: str) -> None:
    """ Check that the regime is valid. """
    if regime not in [NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME]:
        raise ValueError(f'Invalid regime "{regime}". Valid regimes are "{NO_REGIME}", "{DIFFUSIVE_REGIME}" and "{BALLISTIC_REGIME}".')

def monoexp(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Return the monoexponential e^(-b*D).
    """

    [b, D] = at_least_1d([b, D])
    S = np.exp(-np.outer(D, b))
    return np.reshape(S, list(D.shape) + [b.size]) # reshape as np.outer flattens D is ndim > 1

def kurtosis(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], K: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Return the kurtosis signal representation.
    """
    
    [b, D, K] = at_least_1d([b, D, K])
    Slin = monoexp(b, D)
    Squad = np.exp(np.reshape(np.outer(D, b)**2, list(D.shape) + [b.size]) * K[..., np.newaxis]/6)
    return Slin * Squad

def sIVIM(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the simplified IVIM (sIVIM) model.
    """
    
    [b, D, f, S0] = at_least_1d([b, D, f, S0])
    return S0[..., np.newaxis] * ((1-f[..., np.newaxis]) * kurtosis(b, D, K) + np.reshape(np.outer(f, b==0), list(f.shape) + [b.size]))

def ballistic(b: npt.NDArray[np.float64], c: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], vd: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the ballistic IVIM model.
    """

    [b, c, D, f, vd, S0] = at_least_1d([b, c, D, f, vd, S0])
    return S0[..., np.newaxis] * ((1-f[..., np.newaxis])*kurtosis(b, D, K) + f[..., np.newaxis]*monoexp(b, Db)*monoexp(c**2, vd**2))

def diffusive(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], Dstar: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the diffusive IVIM model.
    """

    [b, D, f, Dstar, S0] = at_least_1d([b, D, f, Dstar, S0])
    return S0[..., np.newaxis] * ((1-f[..., np.newaxis])*kurtosis(b, D, K) + f[..., np.newaxis]*monoexp(b, Dstar))