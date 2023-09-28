from src.wrappers.OsipiBase import OsipiBase
from src.original.OGC_AmsterdamUMC.LSQ_fitting import fit_least_squares_array


class OGC_AmsterdamUMC_biexp(OsipiBase):
    """
    Bi-exponential fitting algorithm by Oliver Gurney-Champion, Amsterdam UMC
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Oliver Gurney Champion, Amsterdam UMC"
    id_algorithm_type = "Bi-exponential fit"
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

    def __init__(self, bvalues, bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.2, 1.3])):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(OGC_AmsterdamUMC_biexp, self).__init__(bvalues, None, bounds, None)
        self.OGC_algorithm = fit_least_squares_array

    def ivim_fit(self, signals, bvalues=None):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """

        fit_results = self.OGC_algorithm(bvalues, signals)

        D = fit_results[0]
        f = fit_results[1]
        Dstar = fit_results[2]

        return f, Dstar, D