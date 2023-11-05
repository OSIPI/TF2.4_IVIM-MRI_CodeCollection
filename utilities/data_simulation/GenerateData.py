import numpy as np

class GenerateData:
    """
    Generate exponential and linear data
    """
    def __init__(self, operator=None, rng=None):
        """
        Parameters
        ----------
        operator : numpy or torch
            Must provide mathematical operators
        rng : random number generator
            Must provide normal() operator
        """
        if operator is None:
            self._op = np
        else:
            self._op = operator
        if rng is None:
            self._rng = self._op.random
        else:
            self._rng = rng

    def ivim_signal(self, D, Dp, f, S0, bvalues, snr=None, rician_noise=False):
        """
        Generates IVIM (biexponential) signal

        Parameters
        ----------
        D : float
            The tissue diffusion value
        Dp : float
            The pseudo perfusion value
        f : float
            The fraction of the signal from perfusion
        S0 : float
            The baseline signal (magnitude at no diffusion)
        bvalues : list or array of float
            The diffusion (b-values)
        """
        signal = self.multiexponential_signal([D, Dp], [1 - f, f], S0, self._op.asarray(bvalues, dtype='float64'))
        return self.add_noise(signal, snr, rician_noise)

    def exponential_signal(self, D, bvalues):
        """
        Generates exponential signal

        Parameters
        ----------
        D : float
            The tissue diffusion value
        bvalues : list or array of float
            The diffusion (b-values)
        """
        assert D >= 0, 'D must be >= 0'
        return self._op.exp(-self._op.asarray(bvalues, dtype='float64') * D)

    def multiexponential_signal(self, D, F, S0, bvalues):
        """
        Generates multiexponential signal
        The combination of exponential signals

        Parameters
        ----------
        D : list or arrray of float
            The tissue diffusion value
        F : list or array of float
            The fraction of the signal from perfusion
        S0 : list or array of float
            The baseline signal (magnitude at no diffusion)
        bvalues : list or array of float
            The diffusion (b-values)
        """
        assert len(D) == len(F), 'D and F must be the same length'
        signal = self._op.zeros_like(bvalues)
        for [d, f] in zip(D, F):
            signal += f * self.exponential_signal(d, bvalues)
        signal *= S0
        return signal

    def add_noise(self, real_signal, snr=None, rician_noise=True, imag_signal=None):
        """
        Adds Rician noise to a real or complex signal

        Parameters
        ----------
        real_signal : list or array of float
            The real channel float
        snr : float
            The signal to noise ratio
        imag_signal : list or array of float
            The imaginary channel float
        """
        if imag_signal is None:
            imag_signal = self._op.zeros_like(real_signal)
        real_noise = self._op.zeros_like(real_signal)
        imag_noise = self._op.zeros_like(real_signal)
        if snr is not None:
            real_noise = self._rng.normal(0, 1 / snr, real_signal.shape)
            if rician_noise:
                imag_noise = self._rng.normal(0, 1 / snr, imag_signal.shape)
        noisy_data = self._op.sqrt(self._op.power(real_signal + real_noise, 2) + self._op.power(imag_signal + imag_noise, 2))
        return noisy_data

    def linear_signal(self, D, bvalues, offset=0):
        """
        Generates linear signal

        Parameters
        ----------
        D : float
            The tissue diffusion value
        bvalues : list or array of float
            The diffusion (b-values)
        offset : float
            The signal offset
        """
        assert D >= 0, 'D must be >= 0'
        data = -D * np.asarray(bvalues)
        return data + offset

    def multilinear_signal(self, D, F, S0, bvalues, offset=0):
        """
        Generates multilinear signal
        The combination of multiple linear signals

        Parameters
        ----------
        D : list or arrray of float
            The tissue diffusion value
        F : list or array of float
            The fraction of the signal from perfusion
        S0 : list or array of float
            The baseline signal (magnitude at no diffusion)
        bvalues : list or array of float
            The diffusion (b-values)
        offset : float
            The signal offset
        """
        assert len(D) == len(F), 'D and F must be the same length'
        signal = self._op.zeros_like(bvalues)
        for [d, f] in zip(D, F):
            signal += f * self.linear_signal(d, bvalues)
        signal *= S0
        signal += offset
        return signal