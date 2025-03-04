""" Classes and functions for fitting ivim model """
import numpy as np
from scipy.optimize import shgo
from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.utils.optpkg import optional_package
cvxpy, have_cvxpy, _ = optional_package("cvxpy")

class IvimModelTopoPro(ReconstModel):

    def __init__(self, gtab, bounds=[[0, 0.005, 1e-5], [1, 0.1, 0.004]], \
        rescale_units=False, shgo_iters=5, rescale_results_to_mm2_s=False):
        r""" Initialize an IvimModelTP class.
        This particular script was modified as the DIPY version has stringent 
        hard-coded bounds for the first optimizer that do not work well.
        Note that he returned values are in units of µm2/ms!
        
        gtab : DIPY gtab class containing the b-values
            These are automatically scaled to units of ms/µm2 if given in s/mm2.
        shgo_iters : int, optional
            The number of iterations done by the SHGO optimizer. 5 is recommended.
            Default : 5.
        rescale_units : bool, optional
            Set to True if you are inputting in units of mm2/s and want the
            values automatically rescaled to µm2/ms. The latter is the best
            for this fitting method.
            Default : False
        rescale_results_to_mm2_s : bool, optional
            Set to True if you have used um2/ms but want the results in mm2shgo_iters : int, optional
            The number of iterations done by the SHGO optimizer. 5 is recommended.
            Default : False.
        
        The original script can be found in the authours GitHub
        https://github.com/ShreyasFadnavis/topopro

        The IVIM model assumes that biological tissue includes a volume
        fraction 'f' of water flowing with a pseudo-diffusion coefficient
        D* and a fraction (1-f: treated as a separate fraction in the variable
        projection method) of static (diffusion only), intra and
        extracellular water, with a diffusion coefficient D. In this model
        the echo attenuation of a signal in a single voxel can be written as

            .. math::

            S(b) = S_0*[f*e^{(-b*D\*)} + (1-f)e^{(-b*D)}]

            Where:
            .. math::

            S_0, f, D\* and D are the IVIM parameters.

        References
        ----------
        .. [1] Le Bihan, Denis, et al. "Separation of diffusion and perfusion
               in intravoxel incoherent motion MR imaging." Radiology 168.2
               (1988): 497-505.
        .. [2] Federau, Christian, et al. "Quantitative measurement of brain
               perfusion with intravoxel incoherent motion MR imaging."
               Radiology 265.3 (2012): 874-881.
        .. [3] Fadnavis, Shreyas et.al. "MicroLearn: Framework for machine
               learning, reconstruction, optimization and microstructure
               modeling, Proceedings of: International Society of Magnetic
               Resonance in Medicine (ISMRM), Montreal, Canada, 2019.
        """

        self.bvals = gtab.bvals
        self.yhat_perfusion = np.zeros(self.bvals.shape[0])
        self.yhat_diffusion = np.zeros(self.bvals.shape[0])
        self.exp_phi = np.zeros((self.bvals.shape[0], 2))
        self.shgo_iters = shgo_iters
        self.rescale_results_to_mm2_s = rescale_results_to_mm2_s
        
        # The rescaled units arguement converts the bounds for D* and D from
        # mm2/s to µm2/ms.
        # It is assumed that the b-vales are given in the corresponding units
        # and are thus not changed.
        if gtab.bvals[-1] >= 10:
            self.bvals = gtab.bvals/1000
        if bounds is None:
            # Bounds expressed as (lower bound, upper bound) for [f, D*, D].
            self.bounds = np.array([(0, 1), (5, 100), (0, 4)])
        elif (bounds[0][1] <= 1) or rescale_units: # Realistically, if mm2/s units are used, D* bound is <= 1
            self.bounds = np.array([(bounds[0][0], bounds[1][0]), \
                                    (bounds[0][1]*1000, bounds[1][1]*1000), \
                                    (bounds[0][2]*1000, bounds[1][2]*1000)])
        else: # Finally, if units if µm2/ms are already used
            self.bounds = np.array([(bounds[0][0], bounds[1][0], \
                                    (bounds[0][1], bounds[1][1]), \
                                    (bounds[0][2], bounds[1][2]))])

    @multi_voxel_fit
    def fit(self, data):
        r""" Fit method of the IvimModelTopoPro model class

        Separable Homological Optimization for IVIM [1]_.

        The TopoPro computes the IVIM parameters using the a bi-level
        topological approach. This algorithm uses three different optimizers.
        Level 1: It starts with a Simplicial Homolgy Optimization algorithm and
        fits the parameters in the power of exponentials. Then the fitted
        parameters in the first step are utilized to make a linear convex
        problem. Using a convex optimization, the volume fractions are
        determined.

        Level 2: Simplicial Homolgy Optimization fitting on all the
        parameters. The results of `Level 1` are utilized as
        the initial values for the `Level 2` of the algorithm.

        References
        ----------
        .. [1] Endres, Stefan et.al. "A simplicial homology algorithm for
               Lipschitz optimisation", Journal of Global Optimization, 2018.
        .. [2] Fadnavis, Shreyas et.al. "MicroLearn: Framework for machine
               learning, reconstruction, optimization and microstructure
               modeling, Proceedings of: International Society of Magnetic
               Resonance in Medicine (ISMRM), Montreal, Canada, 2019.
        .. [3] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).

        """

        data_max = data.max()
        if data_max == 0:
            pass
        else:
            data = data / data_max
        b = self.bvals

        # Setting up the bounds for level-1 SHGO
        bounds_sh = np.array(self.bounds[1:])

        # Optimizer #1: SHGO
        minimizer_kwargs_pre = {'options': {f'ftol': 1e-4},
                                'method': 'SLSQP'}
        res_one = shgo(self.stoc_search_cost, bounds_sh, iters=self.shgo_iters,
                       sampling_method='simplicial', args=(data,))
        x = res_one.x
        phi = self.phi(x)

        # Optimizer #2: Convex Optimizer
        f = self.cvx_fit(data, phi)
        x_f = self.x_and_f_to_x_f(x, f)

        # Setting up the bounds for level-2 SHGO
        bounds_simpl = [(x_f[0] - x_f[0]*.99, x_f[0] + x_f[0]*.99),
                        (x_f[1] - x_f[1]*.7, x_f[1] + x_f[1]*.7),
                        (x_f[2] - x_f[2]*.7, x_f[2] + x_f[2]*.7)]

        # build simplex around x_f (bounds must be symmetric)
        minimizer_kwargs = {'options': {f'ftol': 1e-4}}
        res = shgo(self.nlls_cost, bounds_simpl, iters=self.shgo_iters,
                   minimizer_kwargs=minimizer_kwargs,
                   sampling_method='simplicial',
                   args=(data,))

        result = res.x
        f_est = result[0]
        D_star_est = result[1]
        D_est = result[2]

        S0 = data / (f_est * np.exp(-b * D_star_est) + (1 - f_est) *
                     np.exp(-b * D_est))
        S0_est = S0 * data_max

        # final result containing the four fit parameters: S0, f, D* and D
        if self.rescale_results_to_mm2_s:
            result = np.array([np.mean(S0_est), f_est, D_star_est*1e-3, D_est*1e-3])
        else:
            result = np.insert(result, 0, np.mean(S0_est), axis=0)
            
        return IvimFit(self, result)
    
    def rescale_bounds_and_initial_guess(self, rescale_units):
        if rescale_units:
            # Rescale the guess
            self.initial_guess = (self.initial_guess[0], self.initial_guess[1], \
                self.initial_guess[2]*1000, self.initial_guess[3]*1000)
            
            # Rescale the bounds
            lower_bounds = (self.bounds[0][0], self.bounds[0][1], \
                self.bounds[0][2]*1000, self.bounds[0][3]*1000)
            upper_bounds = (self.bounds[1][0], self.bounds[1][1], \
                self.bounds[1][2]*1000, self.bounds[1][3]*1000)
            self.bounds = (lower_bounds, upper_bounds)
            
    def stoc_search_cost(self, x, signal):
        """
        Cost function for SHGO algorithm. Performs an approximation of the
        homology groups of a complex built on a hypersurface homeomorphic to a
        complex on the objective function for the non-linear parameters 'x'.
        The objective funtion is calculated in the :func: `ivim_shgo`.
        The function constructs the parameters using :func: `phi`.

        Parameters
        ----------
        x : array
            input from the Simplicial Homology optimizer.

        signal : array
            The signal values measured for this model.

        Returns
        -------
        :func: `ivim_shgo`

        """
        phi = self.phi(x)

        return self.ivim_shgo(phi, signal)

    def ivim_shgo(self, phi, signal):
        """
        Constructs the objective for the :func: `stoc_search_cost`.

        First calculates the Moore-Penrose inverse of the input `phi` and takes
        a dot product with the measured signal. The result obtained is again
        multiplied with `phi` to complete the projection of the variable into
        a transformed space. (see [1]_ and [2]_ for thorough discussion on
        Variable Projections and relevant cost functions).

        Parameters
        ----------
        phi : array
            Returns an array calculated from :func: `Phi`.

        signal : array
            The signal values measured for this model.

        Returns
        -------
        (signal -  S)^T(signal -  S)

        Notes
        --------
        to make cost function for Differential Evolution algorithm:
        .. math::

            (signal -  S)^T(signal -  S)

        References
        ----------
        .. [1] Fadnavis, Shreyas et.al. "MicroLearn: Framework for machine
               learning, reconstruction, optimization and microstructure
               modeling, Proceedings of: International Society of Magnetic
               Resonance in Medicine (ISMRM), Montreal, Canada, 2019.
        .. [2] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).

        """
        # Moore-Penrose
        phi_mp = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)
        f = np.dot(phi_mp, signal)
        yhat = np.dot(phi, f)
        return np.dot((signal - yhat).T, signal - yhat)

    def cvx_fit(self, signal, phi):
        """
        Performs the constrained search for the linear parameters `f` after
        the estimation of `x` is done. Estimation of the linear parameters `f`
        is a constrained linear least-squares optimization problem solved by
        using a convex optimizer from cvxpy. The IVIM equation contains two
        parameters that depend on the same volume fraction. Both are estimated
        as separately in the convex optimizer.

        Parameters
        ----------
        phi : array
            Returns an array calculated from :func: `phi`.

        signal : array
            The signal values measured for this model.

        Returns
        -------
        f1, f2 (volume fractions)

        Notes
        --------
        cost function for differential evolution algorithm:

        .. math::

            minimize(norm((signal)- (phi*f)))
        """
        # Create four scalar optimization variables.
        f = cvxpy.Variable(2)
        constraints = [cvxpy.sum(f) == 1,
                       f[0] >= 1e-7,
                       f[1] >= 1e-7,
                       f[0] <= 0.9,
                       f[1] <= 0.9]

        # Form objective.
        obj = cvxpy.Minimize(cvxpy.sum(cvxpy.square(phi @ f - signal)))

        # Form and solve problem.
        prob = cvxpy.Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        return np.array(f.value)

    def nlls_cost(self, x_f, signal):
        """
        Cost function for the least square problem. The cost function is used
        in the `Level 2` of TopoPro algorithm :func: `fit`.

        Parameters
        ----------
        x_f : array
            Contains the parameters 'x' and 'f' combines in the same array.

        signal : array
            The signal values measured for this model.

        Returns
        -------
        sum{(signal -  phi*f)^2}

        Notes
        --------
        cost function for the least square problem.

        .. math::

            sum{(signal -  phi*f)^2}
        """
        x, f = self.x_f_to_x_and_f(x_f)
        f1 = np.array([f, 1 - f])
        phi = self.phi(x)

        return np.sum((np.dot(phi, f1) - signal) ** 2)

    def x_f_to_x_and_f(self, x_f):
        """
        Splits the array of parameters in x_f to 'x' and 'f' for performing
        a search on the both of them independently using the simplicial
        homology optimizer (SHGO).

        Parameters
        ----------
        x_f : array
            Combined array of parameters 'x' and 'f' parameters.

        Returns
        -------
        x, f : array
            Splitted parameters into two separate arrays

        """
        x = np.zeros(2)
        f = x_f[0]
        x = x_f[1:3]
        return x, f

    def x_and_f_to_x_f(self, x, f):
        """
        Combines the array of parameters 'x' and 'f' into x_f for performing
        SHGO on the `Level 2` of the optimization process.

        Parameters
        ----------
         x, f : array
            Splitted parameters into two separate arrays

        Returns
        -------
        x_f : array
            Combined array of parameters 'x' and 'f' parameters.

        """
        x_f = np.zeros(3)
        x_f[0] = f[0]
        x_f[1:3] = x
        return x_f

    def phi(self, x):
        """
        Creates a structure for the combining the diffusion and pseudo-
        diffusion by multipling with the bvals and then exponentiating each of
        the two components for fitting as per the IVIM- two compartment model.

        Parameters
        ----------
         x : array
            input from the Differential Evolution optimizer.

        Returns
        -------
        exp_phi1 : array
            Combined array of parameters perfusion/pseudo-diffusion
            and diffusion parameters.

        """
        self.yhat_perfusion = self.bvals * x[0]
        self.yhat_diffusion = self.bvals * x[1]
        self.exp_phi[:, 0] = np.exp(-self.yhat_perfusion)
        self.exp_phi[:, 1] = np.exp(-self.yhat_diffusion)
        return self.exp_phi


class IvimFit(object):

    def __init__(self, model, model_params):
        """ Initialize a IvimFit class instance.
            Parameters
            ----------
            model : Model class
            model_params : array
            The parameters of the model. In this case it is an
            array of ivim parameters. If the fitting is done
            for multi_voxel data, the multi_voxel decorator will
            run the fitting on all the voxels and model_params
            will be an array of the dimensions (data[:-1], 4),
            i.e., there will be 4 parameters for each of the voxels.
        """
        self.model = model
        self.model_params = model_params

    def __getitem__(self, index):
        model_params = self.model_params
        N = model_params.ndim
        if type(index) is not tuple:
            index = (index,)
        elif len(index) >= model_params.ndim:
            raise IndexError("IndexError: invalid index")
        index = index + (slice(None),) * (N - len(index))
        return type(self)(self.model, model_params[index])

    @property
    def S0_predicted(self):
        return self.model_params[..., 0]

    @property
    def perfusion_fraction(self):
        return self.model_params[..., 1]

    @property
    def D_star(self):
        return self.model_params[..., 2]

    @property
    def D(self):
        return self.model_params[..., 3]

    @property
    def shape(self):
        return self.model_params.shape[:-1]
