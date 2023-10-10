# Non-osipi dependencies

# osipi utilities

# osipi implementations

import numpy as np

# from utils.data_simulation.GenerateData import GenerateData
from src.original.ETP_SRI.LinearFitting import LinearFit

author_lookup = {
    'ETP_SRI': LinearFit
}

# print('hi!')



def ivim_fit(authors=None, signals=None, bvalues=None, data=None, initial_guess=None, bounds=None):
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
    if data is not None:
        bvalues = data.bvalues
        signals = data.signals
    

    if bvalues is None:
        print('Missing required argument bvalues')
        return

    if signals is None:
        print('Missing required argument signals')
        return


    # Some implementations can only fit a voxel at a time (i.e. all inputs must be 2-dimensional)
    requires_2D = True if author in [] else False
    requires_4D = True if author in [] else False

    # Bounds and initial guess for parameters
    if initial_guess is None:
        initial_guess = []

    if bounds is None:
        bounds = []

    # Create a fitting function for the chosen author/implementation
    if authors is None:
        print('Missing required argument authors')
    
    for author in authors:
        if author not in author_lookup:
            print(f'Unknown author {author}')
            continue
        print(f'author {author} accepts dimensions {author_lookup[author].accepted_dimensions()}')




if __name__ == '__main__':
    ivim_fit()