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
        assert np.all(D >= 0), 'all values in D must be >= 0'
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
        assert np.all(D >= 0), 'every value in D must be >= 0'
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

    def simulate_training_data(self, bvalues, SNR = (5,100), n = 1000000, Drange = (0.0005,0.0034), frange = (0,1), Dprange = (0.005,0.1), rician_noise = False):
        """
         Simulates IVIM (Intravoxel Incoherent Motion) training data with optional Rician noise.

         Parameters:
         ----------
         bvalues : array-like
             A list or array of b-values used in the diffusion signal simulation.
         SNR : float, or tuple of two floats, optional
             Signal-to-noise ratio. If a tuple (min, max) is provided, SNRs are sampled
             logarithmically between those bounds. If set to 0, no noise is added. Default is (5, 100).
         n : int, optional
             Number of simulated voxels/signals to generate. Default is 1,000,000.
         Drange : tuple of two floats, optional
             Range (min, max) for the diffusion coefficient D (in mm²/s). Default is (0.0005, 0.0034).
         frange : tuple of two floats, optional
             Range (min, max) for the perfusion fraction f. Default is (0, 1).
         Dprange : tuple of two floats, optional
             Range (min, max) for the pseudo-diffusion coefficient Dp (in mm²/s). Default is (0.005, 0.1).
         rician_noise : bool, optional
             If True, Rician noise is added to the simulated signal. Default is False.

         Returns:
         -------
         data_sim : ndarray of shape (n, len(bvalues))
             Simulated IVIM signal data normalized by S0.
         D : ndarray of shape (n, 1)
             Ground truth diffusion coefficient values used in simulation.
         f : ndarray of shape (n, 1)
             Ground truth perfusion fraction values used in simulation.
         Dp : ndarray of shape (n, 1)
             Ground truth pseudo-diffusion coefficient values used in simulation.

         Notes:
         -----
         - The function uses self.ivim_signal() to generate signals.
         - Noise is applied after generating noise-free IVIM signals, using either Gaussian or Rician noise.
         - Simulated signals are normalized by the mean S0 (b = 0) signal.
         """
        test = self._rng.uniform(0, 1, (n, 4))
        D = Drange[0] + test[:, [0]] * (Drange[1] - Drange[0])
        f = frange[0] + test[:, [1]] * (frange[1] - frange[0])
        Dp = Dprange[0] + test[:, [2]] * (Dprange[1] - Dprange[0])
        #data_sim = np.zeros([len(D), len(bvalues)])
        bvalues = np.array(bvalues)
        if type(SNR) == tuple:
            noise_std = 1/SNR[1] + test[:,3] * (1/SNR[0] - 1/SNR[1])
            addnoise = True
        elif SNR == 0:
            addnoise = False
            noise_std = np.ones((n, 1))
        else:
            noise_std = np.full((n, 1), 1/SNR)
            addnoise = True
        noise_std = noise_std[:, np.newaxis]
        # loop over array to fill with simulated IVIM data
        bvalues = np.array(bvalues).reshape(1, -1)
        data_sim = 1 * (f * np.exp(-bvalues * Dp) + (1 - f) * np.exp(-bvalues * D))

        # if SNR is set to zero, don't add noise
        if addnoise:
            noise_real = self._rng.normal(0, noise_std, data_sim.shape)
            noise_imag = self._rng.normal(0, noise_std, data_sim.shape)

            if rician_noise:
                data_sim = np.sqrt((data_sim + noise_real) ** 2 + noise_imag ** 2)
            else:
                data_sim = data_sim + noise_real
        S0_noisy = np.mean(data_sim[:, bvalues.flatten() == 0], axis=1)
        data_sim = data_sim / S0_noisy[:, None]
        return data_sim, D, f, Dp

