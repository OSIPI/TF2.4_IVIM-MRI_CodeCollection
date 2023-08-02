


class OsipiBase:
    """The base class for OSIPI IVIM fitting"""
    
    #def __init__(self, author, data_dimension, thresholds_required, guess_required, bounds_required):
    #    pass

    def fit_osipi(self, data=None, b_values=None, initial_guess=None, bounds=None, **kwargs):
        """Fits the data with the bvalues
        Returns [S0, f, D*, D]
        """
        #self.parameter_estimates = self.ivim_fit(data, b_values)
        pass

    def accepted_dimensions_osipi(self):
        """The array of accepted dimensions
        e.g.
        (1D,   2D,   3D,    4D,    5D,    6D)
        (True, True, False, False, False, False)
        """
        return (False,) * 6

    def accepts_dimension_osipi(self, dim):
        """Query if the selection dimension is fittable"""
        accepted = self.accepted_dimensions()
        if dim < 0 or dim > len(accepted):
            return False
        return accepted[dim]
    
    def thresholds_required_osipi():
        """How many segmentation thresholds does it require?"""
        return 0

    def guess_required_osipi():
        """Does it require an initial guess?"""
        return False

    def bounds_required_osipi():
        """Does it require bounds?"""
        return False

    def author_osipi():
        """Author identification"""
        return ''
    
    def simple_test():
        pass
    
