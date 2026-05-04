import numpy as np
import numpy.typing as npt

def seg(Y, b, bthr = 200, verbose = False):
    """
    Segmented fitting of the IVIM model.

    Arguments:
        Y:         v x b matrix with data
        b:         vector of size b with b-values 
        bthr:      (optional) threshold b-value from which signal is included in first fitting step
        verbose:   (optional) if True, diagnostics during fitting is printet to terminal
    """

    def valid_signal(Y):
        """
        Return a mask representing all rows in Y with valid values (not non-positive, NaN or infinite).

        Arguments:
        Y    -- v x b matrix with data

        Output:
        mask -- vector of size v indicating valid rows in Y
        """

        mask = ~np.any((Y<=0) | np.isnan(Y) | np.isinf(Y), axis=1)
        return mask
    
    def monoexp_model(b, D):
        """
        Return the monoexponential e^(-b*D).
        
        Arguments:
            b: vector of b-values [s/mm2]
            D: ND array of diffusion coefficients [mm2/s]

        Output:
            S: (N+1)D array of signal values
        """
        b = np.atleast_1d(b)
        D = np.atleast_1d(D)
        S = np.exp(-np.outer(D, b))
        return np.reshape(S, list(D.shape) + [b.size]) # reshape as np.outer flattens D is ndim > 1


    def _monoexp(Y, b, lim = [0, 3e-3], validate = True, verbose = False):
        """ Estimate D and A (y axis intercept) from a monoexponential model. """

        if validate:
            mask = valid_signal(Y) # Avoids signal with obvious errors (non-positive, nan, inf)
        else:
            mask = np.full(Y.shape[0], True)
            
        D = np.full(mask.shape, np.nan)
        A = np.full(mask.shape, np.nan)

        if b.size == 2:
            if b[1] == b[0]:
                raise ZeroDivisionError("Two b-values can't be equal or both zero.")
            D[mask] = (np.log(Y[mask, 0]) - np.log(Y[mask, 1])) / (b[1] - b[0])
            
            D[mask & (D<lim[0])] = lim[0]
            D[mask & (D>lim[1])] = lim[1]

            A[mask] = Y[mask, 0] * np.exp(b[0]*D[mask])
        elif b.size > 2:
            D[mask], A[mask] = _optimizeD(Y[mask, :], b, lim, disp_prog = verbose)
        else:
            raise ValueError('Too few b-values.')

        return D, A

    def _f_from_intercept(A, S0):
        """ Calculate f from S(b=0) and extrapolated y axis intercept A."""
        f = 1 - A/S0
        f[f<0] = 0
        return f
    
    def _optimizeD(Y, b, lim, optlim = 1e-6, disp_prog = False):
        """ Specfically tailored function for finding the least squares solution of monoexponenital fit. """

        n = Y.shape[0]
        D = np.zeros(n)
        yb = Y * np.tile(b, (n, 1))  # Precalculate for speed.
        
        ##############################################
        # Check if a minimum is within the interval. #
        ##############################################
        # Check that all diff < 0 for Dlow.
        Dlow = lim[0] * np.ones(n)
        difflow,_ = _Ddiff(Y, yb, b, Dlow)
        low_check = difflow < 0 # difflow must be < 0 if the optimum is within the interval.
        
        # Check that all diff > 0 for Dhigh
        Dhigh = lim[1] * np.ones(n)
        diffhigh,_ = _Ddiff(Y, yb, b, Dhigh)
        high_check = diffhigh > 0  # diffhigh must be > 0 if the optimum is within the interval.
        
        # Set parameter value with optimum out of bounds.
        D[~low_check] = lim[0]  # difflow > 0 means that the mimimum has been passed .
        D[~high_check] = lim[1]  # diffhigh < 0 means that the minium is beyond the interval.
        
        # Only the voxels with a possible minimum should be estimated.
        mask = low_check & high_check
        if disp_prog:
            print(f'Discarding {np.count_nonzero(~mask)} voxels due to parameters out of bounds.')

        # Allocate all variables.
        D_lin = np.zeros(n)
        diff_lin = np.zeros(n)
        D_mid = np.zeros(n)
        diff_mid = np.zeros(n)
        ratio_lin = np.zeros(n)
        ratio_mid = np.zeros(n)

        ##########################################################
        # Iterative method for finding the point where diff = 0. #
        ##########################################################
        k = 0
        while np.any(mask):  # Continue if there are voxels left to optimize.
            # Assume diff is linear within the search interval [Dlow Dhigh].
            D_lin[mask] = Dlow[mask] - difflow[mask] * (Dhigh[mask]-Dlow[mask]) / (diffhigh[mask]-difflow[mask])
            # Calculate diff in the point of intersection given by the previous expression.
            diff_lin[mask], ratio_lin[mask] = _Ddiff(Y[mask, :], yb[mask, :], b, D_lin[mask])
        
            # As a potential speed up, the mean of Dlow and Dhigh is also calculated.
            D_mid[mask] = (Dlow[mask]+Dhigh[mask]) / 2
            diff_mid[mask], ratio_mid[mask] = _Ddiff(Y[mask, :], yb[mask, :], b, D_mid[mask])
            
            # If diff < 0, then the point of intersection or mean is used as the
            # new Dlow. Only voxels with diff < 0 are updated at this step. Linear
            # interpolation or the mean is used depending of which method that
            # gives the smallest diff.
            updatelow_lin = (diff_lin<0) & ((diff_mid>0) | ((D_lin>D_mid) & (diff_mid<0)))
            updatelow_mid = (diff_mid<0) & ((diff_lin>0) | ((D_mid>D_lin) & (diff_lin<0)))
            Dlow[updatelow_lin] = D_lin[updatelow_lin]
            Dlow[updatelow_mid] = D_mid[updatelow_mid]
            
            # If diff > 0, then the point of intersection or mean is used as the
            # new Dhigh. Only voxels with diff > 0 are updated at this step. 
            # Linear interpolation or the mean is used depending of which method 
            # that gives the smallest diff.
            updatehigh_lin = (diff_lin>0) & ((diff_mid<0) | ((D_lin<D_mid) & (diff_mid>0)))
            updatehigh_mid = (diff_mid>0) & ((diff_lin<0) | ((D_mid<D_lin) & (diff_lin>0)))
            Dhigh[updatehigh_lin] = D_lin[updatehigh_lin]
            Dhigh[updatehigh_mid] = D_mid[updatehigh_mid]
            
            # Update the mask to exclude voxels that fulfills the optimization
            # limit from the mask.
            opt_lin = np.abs(1-ratio_lin) < optlim
            opt_mid = np.abs(1-ratio_mid) < optlim
            
            D[opt_lin] = D_lin[opt_lin]
            D[opt_mid] = D_mid[opt_mid]  
            # Not optimal if both D_lin and D_mean fulfills the optimization limit,
            # but has a small impact on the result as long as optlim is small.
            
            # Update the mask.
            mask = mask & (~(opt_lin|opt_mid))
            
            # Calculate diff for the new bounds.
            if np.any(mask):
                difflow[mask],_ = _Ddiff(Y[mask, :], yb[mask, :], b, Dlow[mask])
                diffhigh[mask],_ = _Ddiff(Y[mask, :], yb[mask, :], b, Dhigh[mask])
            
            k += 1
            if disp_prog:
                print(f'Iteration {k}: {np.count_nonzero(mask)} voxels left.')

        A = np.sum(Y*np.exp(-np.outer(D, b)), axis=1) / np.sum(np.exp(-2*np.outer(b, D)), axis=0)

        return D, A


    def _Ddiff(Y, yb, b, D):
        """
        Return the difference between q1 = e^(-2*b*D)*yb*e^(-b*D) and 
        q2 = Y*e^(-b*D)*b*e^(-2*b*D) summed over b as well as the ratio q1/q2
        summed over b, setting divisions by zero as infinite.
        """
        q1 = np.sum(np.exp(-2*np.outer(b, D)), axis=0) * np.sum(yb*np.exp(-np.outer(D, b)), axis=1)
        q2 = np.sum(Y*np.exp(-np.outer(D, b)), axis=1) * np.sum(b[:, np.newaxis]*np.exp(-2*np.outer(b, D)), axis=0)
        diff = q1 - q2
        ratio = np.full(q1.shape, np.inf)
        ratio[q2!=0] = q1[q2!=0] / q2[q2!=0]
        return diff, ratio

    if b.ndim != 1:
        raise ValueError('b must a 1D array')

    if Y.ndim != 2:
        if Y.ndim != 1:
            raise ValueError('Y must be a 2D array')
        else:
            Y = Y[np.newaxis,:]

    if b.size != Y.shape[1]:
        raise ValueError('Number of b-values must match the second dimension of Y.')

    bmask = b >= bthr

    D, A = _monoexp(Y[:, bmask], b[bmask], verbose=verbose)
    Ysub = (Y - A[:, np.newaxis]*monoexp_model(b, D))        # Remove signal related to diffusion

    Dstar, Astar = _monoexp(Ysub, b, lim=[3e-3, 0.1], validate = False, verbose = verbose)
    S0 = A + Astar

    f = _f_from_intercept(A, S0)

    if Y.shape[0] == 1:
        D = D.item()
        f = f.item()
        Dstar = Dstar.item()
        S0 = S0.item()

    pars = {'D': D, 'f': f, 'Dstar': Dstar, 'S0': S0}
    return pars