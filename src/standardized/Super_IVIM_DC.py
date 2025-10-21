from src.wrappers.OsipiBase import OsipiBase
import numpy as np
import os
from super_ivim_dc.train import train
from pathlib import Path
from super_ivim_dc.infer import infer_from_signal
import warnings


class Super_IVIM_DC(OsipiBase):
    """
    Supervised deep learnt fitting algorithm by Moti Freiman and Noam Korngut, TechnionIIT
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Moti Freiman and Noam Korngut, TechnIIT"
    id_algorithm_type = "Supervised Deep learnt bi-exponential fit with data consistency"
    id_return_parameters = "f, D*, D, S0"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"

    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0,
                           0]  # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True  # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = True
    accepted_dimensions = 1  # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?


    # Supported inputs in the standardized class
    supported_bounds = True
    supported_initial_guess = True
    supported_thresholds = False

    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, fitS0=True, SNR = None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        if bvalues is None:
            raise ValueError("for deep learning models, bvalues need defining at initiaition")
        super(Super_IVIM_DC, self).__init__(bvalues=bvalues, bounds=bounds, initial_guess=initial_guess)
        self.fitS0=fitS0
        self.bvalues=np.array(bvalues)
        self.initialize(bounds, initial_guess, fitS0, SNR)

    def initialize(self, bounds, initial_guess, fitS0, SNR, working_dir=os.getcwd(),ivimnet_filename='ivimnet',super_ivim_dc_filename='super_ivim_dc'):
        if SNR is None:
            warnings.warn('No SNR indicated. Data simulated with SNR = 100')
            SNR=100
        self.fitS0=fitS0
        self.use_initial_guess = False
        self.use_bounds = False
        self.deep_learning = True
        self.supervised = True

        # Additional options
        self.stochastic = True

        modeldir = Path(working_dir)  # Ensure it's a Path object
        modeldir = modeldir / "models"
        modeldir.mkdir(parents=True, exist_ok=True)
        working_dir: str = str(modeldir)
        super_ivim_dc_filename: str = super_ivim_dc_filename  # do not include .pt
        ivimnet_filename: str = ivimnet_filename  # do not include .pt
        self.super_ivim_dc_filename=super_ivim_dc_filename
        self.ivimnet_filename=ivimnet_filename
        self.working_dir=working_dir
        train(
            SNR=SNR,
            bvalues=self.bvalues,
            super_ivim_dc=True,
            work_dir=self.working_dir,
            super_ivim_dc_filename=self.super_ivim_dc_filename,
            ivimnet_filename=ivimnet_filename,
            verbose=False,
            ivimnet=False
        )


    def ivim_fit(self, signals, bvalues, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            results: a dictionary containing "d", "f", and "Dp".
        """
        if not np.array_equal(bvalues, self.bvalues):
            raise ValueError("bvalue list at fitting must be identical as the one at initiation, otherwise it will not run")
        if np.shape(np.shape(signals)) == (1,):
            signals=signals[np.newaxis, :]
        Dp, Dt, f, S0_superivimdc = infer_from_signal(
            signal=signals,
            bvalues=self.bvalues,
            model_path=f"{self.working_dir}/{self.super_ivim_dc_filename}.pt",
        )

        Dp = float(np.atleast_1d(Dp).ravel()[0])
        Dt = float(np.atleast_1d(Dt).ravel()[0])
        f = float(np.atleast_1d(f).ravel()[0])

        results = {}
        results["D"] = safe_scalar(float(np.array(Dt).item()))
        results["f"] = safe_scalar(float(np.array(f).item()))
        results["Dp"] = safe_scalar(float(np.array(Dp).item()))

        return results


    def ivim_fit_full_volume(self, signals, bvalues, retrain_on_input_data=False, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """
        if not np.array_equal(bvalues, self.bvalues):
            raise ValueError("bvalue list at fitting must be identical as the one at initiation, otherwise it will not run")

        nanmask = np.any(np.isnan(signals),axis=-1)
        signals,shape = self.reshape_to_voxelwise(signals)
        Dp, Dt, f, S0_superivimdc = infer_from_signal(
            signal=signals,
            bvalues=self.bvalues,
            model_path=f"{self.working_dir}/{self.super_ivim_dc_filename}.pt",
        )

        results = {}
        for name,par in zip(["D","f","Dp"],[Dt,f,Dp]):
            parmap = np.full(shape[:-1],np.nan)
            parmap[~nanmask] = par
            results[name] = parmap
#        results["f"] = np.reshape(f,shape[:-1])
 #       results["Dp"] = np.reshape(Dp,shape[:-1])

        return results


    def reshape_to_voxelwise(self, data):
        """
        reshapes multi-D input (spatial dims, bvvalue) data to 2D voxel-wise array
        Args:
            data (array): mulit-D array (data x b-values)
        Returns:
            out (array): 2D array (voxel x b-value)
        """
        B = data.shape[-1]
        voxels = int(np.prod(data.shape[:-1]))  # e.g., X*Y*Z
        return data.reshape(voxels, B), data.shape

def safe_scalar(val):
    """Ensure val is always a Python float scalar."""
    try:
        return float(np.array(val).ravel()[0])
    except Exception:
        return 0.0  # fallback if conversion fails