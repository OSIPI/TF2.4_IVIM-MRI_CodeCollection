import numpy as np
import numpy.polynomial.polynomial as poly

from utilities.data_simulation.GenerateData import GenerateData

class LinearFit:
    """
    Performs linear fits of exponential data
    """
    def __init__(self, linear_cutoff=500):
        """
        Parameters
        ----------
        linear_cutoff : float
            The b-value after which it can be assumed that the perfusion value is negligible
        """
        self.linear_cutoff = linear_cutoff

    def accepted_dimensions(self):
        return (1, 1)
    
    def linear_fit(self, bvalues, signal, weighting=None, stats=False):
        """
        Fit a single line

        Parameters
        ----------
        bvalues : list or array of float
            The diffusion (b-values)
        signal : list or array of float
            The acquired signal to fit. It is assumed to be linearized at this point.
        weighting : list or array fo float
            Weights to pass into polyfit. None is no weighting.
        stats : boolean
            If true, return the polyfit statistics
        """
        bvalues = np.asarray(bvalues)
        signal = np.asarray(signal)
        assert bvalues.size == signal.size, "Signal and b-values don't have the same number of values"
        if stats:
            D, stats = poly.polyfit(np.asarray(bvalues), signal, 1, full=True, w=weighting)
            return [np.exp(D[0]), *-D[1:]], stats
        else:
            D = poly.polyfit(np.asarray(bvalues), signal, 1, w=weighting)
            return [np.exp(D[0]), *-D[1:]]

    def ivim_fit(self, bvalues, signal):
        """
        Fit an IVIM curve
        This fits a bi-exponential curve using linear fitting only


        Parameters
        ----------
        bvalues : list or array of float
            The diffusion (b-values)
        signal : list or array of float
            The acquired signal to fit. It is assumed to be exponential at this point
        """
        bvalues = np.asarray(bvalues)
        assert bvalues.size > 1, 'Too few b-values'
        signal = np.asarray(signal)
        assert bvalues.size == signal.size, "Signal and b-values don't have the same number of values"
        gd = GenerateData()
        lt_cutoff = bvalues <= self.linear_cutoff
        gt_cutoff = bvalues >= self.linear_cutoff
        linear_signal = np.log(signal)
        D = self.linear_fit(bvalues[gt_cutoff], linear_signal[gt_cutoff])
        # print(D)
        D[1] = max(D[1], 0)  # constrain to positive values
        
        if lt_cutoff.sum() > 0:
            signal_Dp = linear_signal[lt_cutoff] - gd.linear_signal(D[1], bvalues[lt_cutoff], np.log(D[0]))
            # print(signal_Dp)
            signal_valid = signal_Dp > 0
            lt_cutoff_dual = np.logical_and(lt_cutoff[:len(signal_Dp)], signal_valid)
            # print(lt_cutoff_dual)
            Dp_prime = [-1, -1]
            if lt_cutoff_dual.sum() > 0:
                # print(np.log(signal_Dp[lt_cutoff_dual]))
                Dp_prime = self.linear_fit(bvalues[:len(signal_Dp)][lt_cutoff_dual], np.log(signal_Dp[lt_cutoff_dual]))
                # print(Dp_prime)
            
            if np.any(np.asarray(Dp_prime) < 0) or not np.all(np.isfinite(Dp_prime)):
                print('Perfusion fit failed')
                Dp_prime = [0, 0]
            f = signal[0] - D[0]
        else:
            print("This doesn't seem to be an IVIM set of b-values")
            f = 1
            Dp_prime = [0, 0]
        D = D[1]
        Dp = D + Dp_prime[1]
        if np.allclose(f, 0):
            Dp = 0
        elif np.allclose(f, 1):
            D = 0
        return [f, D, Dp]
