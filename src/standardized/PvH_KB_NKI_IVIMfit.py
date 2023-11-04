from src.wrappers.OsipiBase import OsipiBase
from src.original.PvH_KB_NKI.DWI_functions_standalone import generate_IVIMmaps_standalone, generate_ADC_standalone
import numpy as np

class PvH_KB_NKI_IVIMfit(OsipiBase):
    """
    Bi-exponential fitting algorithm by Petra van Houdt and Koen Baas, NKI
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Group Uulke van der Heide, NKI"
    id_algorithm_type = "Bi-exponential fit"
    id_return_parameters = "f, D*, D"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"

    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0,
                           0]  # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = False  # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional =False
    accepted_dimensions = 1  # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?

    def __init__(self, bvalues=None, thresholds=None,bounds=None,initial_guess=None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(PvH_KB_NKI_IVIMfit, self).__init__(bvalues, thresholds,bounds,initial_guess)
        self.NKI_algorithm = generate_IVIMmaps_standalone


    def ivim_fit(self, signals, bvalues=None):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """
        #bvalues = np.array(bvalues)
        bvalues = bvalues.tolist() #NKI code expects a list instead of nparray
        # reshape signal as the NKI code expects a 4D array
        signals = np.reshape(signals, (1, 1, 1, len(signals)))  # assuming that in this test the signals are always single voxel
        fit_results = self.NKI_algorithm(signals,bvalues)

        D = fit_results[0][0,0,0]/1000
        f = fit_results[1][0,0,0]
        Dstar = fit_results[2][0,0,0]/1000

        return f, Dstar, D
