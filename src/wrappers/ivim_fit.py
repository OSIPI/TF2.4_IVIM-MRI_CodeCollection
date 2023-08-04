# Non-osipi dependencies

# osipi utilities

# osipi implementations


def ivim_fit(author, signals=None, bvalues=None, data=None, initial_guess=None, bounds=None):
    """
    wrapper function to use OSIPI code contributions for IVIM fit
    :param author: str, can be one of []
    :param signals: numpy array containing signal intensities
    :param bvalues: numpy array containing corresponding b-values
    :param data: object containing signals and bvalues
    :param initial_guess: list of initial parameter estimates
    :param bounds: list containing list of lower parameter bounds and list of upper parameter bounds
    :return: numpy array of shape (signals.size, 4) with the D, Dp, f, S0 per voxel.
    """

    # Unpack variables if data object is given
    if not data == None:
        bvalues = data.bvalues
        signals = data.signals


    # Some implementations can only fit a voxel at a time (i.e. all inputs must be 2-dimensional)
    requires_2D = True if author in [] else False
    requires_4D = True if author in [] else False

    # Bounds and initial guess for parameters
    initial_guess = []
    bounds = []

    # Create a fitting function for the chosen author/implementation
    if author == "":
        pass

