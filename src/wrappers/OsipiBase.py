import numpy as np
import importlib
from scipy.stats import norm
import pathlib
import sys
from tqdm import tqdm
from joblib import Parallel, delayed


class OsipiBase:
    """
    Comprehensive base class for OSIPI IVIM fitting algorithms.

    The ``OsipiBase`` class defines a standardized framework for fitting
    intravoxel incoherent motion (IVIM) models to diffusion-weighted MRI data.
    It manages common attributes such as b-values, parameter bounds,
    thresholds, and initial guesses; provides requirement-checking utilities;
    and offers convenience methods for voxel-wise or full-volume fitting.
    Subclasses can implement algorithm-specific behavior while inheriting
    these shared tools.

    Parameters
    ----------
    bvalues : array-like, optional
        Diffusion b-values (s/mm²) matching the last dimension of the input data.
    thresholds : array-like, optional
        Thresholds used by specific algorithms (e.g., signal cutoffs).
    bounds : dict, optional
        Parameter bounds for constrained optimization. Should be a dict with keys
        like "S0", "f", "Dp", "D" and values as [lower, upper] lists or arrays.
        E.g. {"S0" : [0.7, 1.3], "f" : [0, 1], "Dp" : [0.005, 0.2], "D" : [0, 0.005]}.
    initial_guess : dict, optional
        Initial parameter estimates for the IVIM fit. Should be a dict with keys
        like "S0", "f", "Dp", "D" and float values. 
        E.g. {"S0" : 1, "f" : 0.1, "Dp" : 0.01, "D" : 0.001}.
    algorithm : str, optional
        Name of an algorithm module in ``src/standardized`` to load dynamically.
        If supplied, the instance is immediately converted to that algorithm’s
        subclass via :meth:`osipi_initiate_algorithm`.
    force_default_settings : bool, optional
        If bounds and initial guesses are not provided, the wrapper will set
        them to reasonable physical ones, i.e. {"S0":[0.7, 1.3], "f":[0, 1], 
        "Dp":[0.005, 0.2], "D":[0, 0.005]}. 
        To prevent this, set this bool to False. Default initial guess 
        {"S0" : 1, "f": 0.1, "Dp": 0.01, "D": 0.001}.
    **kwargs
        Additional keyword arguments forwarded to the selected algorithm’s
        initializer if ``algorithm`` is provided.

    Attributes
    ----------
    bvalues, thresholds, bounds, initial_guess : np.ndarray or None
        Core fitting inputs stored as arrays when provided.
    use_bounds, use_initial_guess : bool
        Flags controlling whether bounds and initial guesses are applied.
    deep_learning, supervised, stochastic : bool
        Indicators for the algorithm type; subclasses may set these.
    result_keys : list of str, optional
        Names of the output parameters (e.g., ["f", "Dp", "D"]).
    required_bvalues, required_thresholds, required_bounds,
    required_bounds_optional, required_initial_guess,
    required_initial_guess_optional : various, optional
        Optional attributes used by requirement-checking helpers.

    Key Methods
    -----------
    osipi_initiate_algorithm(algorithm, **kwargs)
        Dynamically replace the current instance with the specified
        algorithm subclass.
    osipi_fit(data, bvalues=None, njobs=1, **kwargs)
        Voxel-wise IVIM fitting with optional parallel processing and
        automatic signal normalization.
    osipi_fit_full_volume(data, bvalues=None, **kwargs)
        Full-volume fitting for algorithms that support it.
    osipi_print_requirements()
        Display algorithm requirements such as needed b-values or bounds.
    osipi_accepted_dimensions(), osipi_accepts_dimension(dim)
        Query acceptable input dimensionalities.
    osipi_check_required_*()
        Validate that provided inputs meet algorithm requirements.
    osipi_simple_bias_and_RMSE_test(SNR, bvalues, f, Dstar, D, noise_realizations)
        Monte-Carlo bias/RMSE evaluation of the current fitting method.
    D_and_Ds_swap(results)
        Ensure consistency of D and D* estimates by swapping if necessary.

    Notes
    -----
    * This class is typically used as a base or as a dynamic loader for
      a specific algorithm implementation.
    * Parallel voxel-wise fitting uses :mod:`joblib`.
    * Subclasses must implement algorithm-specific methods such as
      :meth:`ivim_fit` or :meth:`ivim_fit_full_volume`.

    Examples
    --------
    # Dynamic algorithm loading
    model = OsipiBase(bvalues=[0, 50, 200, 800],
                      algorithm="MyAlgorithm")

    # Voxel-wise fitting
    results = base.osipi_fit(dwi_data, njobs=4)
    f_map = results["f"]
    """
    
    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, algorithm=None, force_default_settings=True, **kwargs):
        # Define the attributes as numpy arrays only if they are not None
        self.bvalues = np.asarray(bvalues) if bvalues is not None else None
        self.thresholds = np.asarray(thresholds) if thresholds is not None else None
        self.bounds = bounds if bounds is not None else None
        self.initial_guess = initial_guess if initial_guess is not None else None
        self.use_bounds = True
        self.use_initial_guess = True
        self.deep_learning = False
        self.supervised = False
        self.stochastic = False
        
        if force_default_settings:
            if self.bounds is None:
                print('warning, no bounds were defined, so default bounds are used of [0, 0, 0.005, 0.7],[0.005, 1.0, 0.2, 1.3]')
                self.bounds = {"S0" : [0.7, 1.3], "f" : [0, 1.0], "Dp" : [0.005, 0.2], "D" : [0, 0.005]} # These are defined as [lower, upper]
                self.forced_default_bounds = True
                self.use_bounds = True

            if self.initial_guess is None:
                print('warning, no initial guesses were defined, so default bounds are used of  [0.001, 0.001, 0.01, 1]')
                self.initial_guess = {"S0" : 1, "f" : 0.1, "Dp" : 0.01, "D" : 0.001}
                self.forced_default_initial_guess = True
                self.use_initial_guess = True

        self.osipi_bounds = self.bounds # self.bounds will change form, store it in the osipi format here
        self.osipi_initial_guess = self.initial_guess # self.initial_guess will change form, store it in the osipi format here

        # If the user inputs an algorithm to OsipiBase, it is intereprete as initiating
        # an algorithm object with that name.
        if algorithm:
            self.osipi_initiate_algorithm(algorithm, bvalues=self.bvalues, thresholds=self.thresholds, bounds=self.bounds, initial_guess=self.initial_guess, **kwargs)
            
    def osipi_initiate_algorithm(self, algorithm, **kwargs):
        """Turns the class into a specified one by the input.
        This method can be used instead of specifically importing that class and initiating it.
        
        WIP: Add args and kwargs!

        Args:
            algorithm (string): The name of the algorithm, should be the same as the file in the src/standardized folder without the .py extension.
        """
        
        # Import the algorithm
        root_path = pathlib.Path(__file__).resolve().parents[2]
        if str(root_path) not in sys.path:
            print("Root folder not in PYTHONPATH")
            return False
            
        import_base_path = "src.standardized"
        import_path = import_base_path + "." + algorithm
        #Algorithm = getattr(importlib.import_module(import_path), algorithm)
        # Change the class from OsipiBase to the specified algorithm
        self.__class__ = getattr(importlib.import_module(import_path), algorithm)
        self.__init__(**kwargs)
    
    def initialize(**kwargs):
        """Placeholder for subclass initialization"""
        pass

    #def osipi_fit(self, data=None, bvalues=None, thresholds=None, bounds=None, initial_guess=None, **kwargs):
    def osipi_fit(self, data, bvalues=None, njobs=1, **kwargs):
        """
        Fit multi-b-value diffusion MRI data using the IVIM model.

        This function normalizes the input signal to the minimum b-value and fits each voxel's signal to the IVIM model to estimate parameters such as
        perfusion fraction (f), pseudo-diffusion coefficient (D*), and tissue diffusion coefficient (D).
        Fits can be computed in parallel across voxels using multiple jobs.

        Parameters
        ----------
        data : np.ndarray
            Multi-dimensional array containing the signal intensities. The last dimension must correspond
            to the b-values (diffusion weightings).
        bvalues : array-like, optional
            Array of b-values corresponding to the last dimension of `data`. If not provided, the method
            uses `self.bvalues` set during object initialization.
        njobs : int, optional, default=1
            Number of parallel jobs to use for voxel-wise fitting. If `njobs` > 1, the fitting will be
            distributed across multiple processes. -1 will use all available cpus
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the underlying `ivim_fit` function.

        Returns
        -------
        results : dict of np.ndarray
            Dictionary containing voxel-wise parameter maps. Keys are parameter names ("f", "D", "Dp"),
            and values are arrays with the same shape as `data` excluding the last dimension.

        Notes
        -----
        - The signal is normalized to the minimum b-value before fitting.
        - Handles NaN values by returning zeros for all parameters in those voxels.
        - Parallelization is handled using joblib's `Parallel` and `delayed`.
        - If `self.result_keys` is defined, it determines the output parameter names; otherwise, the default
          keys are ["f", "Dp", "D"].
        - The method swaps D and D* values after fitting using `self.D_and_Ds_swap` to maintain consistency.

        Example
        -------
        >>> results = instance.osipi_fit(data, bvalues=[0, 50, 200, 800], njobs=4)
        >>> f_map = results['f']
        >>> D_map = results['D']
        """
        
        # We should first check whether the attributes in the __init__ are not None
        # Then check if they are input here, if they are, these should overwrite the attributes
        use_bvalues = bvalues if bvalues is not None else self.bvalues
        #use_thresholds = thresholds if self.thresholds is None else self.thresholds
        #use_bounds = bounds if self.bounds is None else self.bounds
        #use_initial_guess = initial_guess if self.initial_guess is None else self.initial_guess

        # Make sure we don't make arrays of None's
        if use_bvalues is not None: use_bvalues = np.asarray(use_bvalues) 
        #if use_thresholds is not None: use_thresholds = np.asarray(use_thresholds) 
        #if use_bounds is not None: use_bounds = np.asarray(use_bounds) 
        #if use_initial_guess is not None: use_initial_guess = np.asarray(use_initial_guess) 
        #kwargs["bvalues"] = use_bvalues
        
        #args = [data, use_bvalues, use_thresholds]
        #if self.required_bounds or self.required_bounds_optional:
            #args.append(use_bounds)
        #if self.required_initial_guess or self.required_initial_guess_optional:
            #args.append(use_initial_guess)
            
        # Run a check_requirements method that makes sure that these inputs fulfil the requirements
        
        
        # Then we check which inputs are required for the algorithm using the requirements attributes in the template 
        
        # Then we pass everything into the ivim_fit method which performs the fit according to the algorithm
        
        #args = [data, use_bvalues, use_initial_guess, use_bounds, use_thresholds]
        #args = [arg for arg in args if arg is not None]

        # Check if there is an attribute that defines the result dictionary keys
        if hasattr(self, "result_keys"):
            # result_keys is a list of strings of parameter names, e.g. "S0", "f1", "f2", etc.
            result_keys = self.result_keys
        else:
            # Default is ["f", "Dp", "D"]
            self.result_keys = ["f", "Dp", "D"]
        
        results = {}
        for key in self.result_keys:
            results[key] = np.empty(list(data.shape[:-1]))

        # Assuming the last dimension of the data is the signal values of each b-value
        # results = np.empty(list(data.shape[:-1])+[3]) # Create an array with the voxel dimensions + the ones required for the fit
        #for ijk in tqdm(np.ndindex(data.shape[:-1]), total=np.prod(data.shape[:-1])):
            #args = [data[ijk], use_bvalues]
            #fit = list(self.ivim_fit(*args, **kwargs))
            #results[ijk] = fit
        minimum_bvalue = np.min(use_bvalues) # We normalize the signal to the minimum bvalue. Should be 0 or very close to 0.
        b0_indices = np.where(use_bvalues == minimum_bvalue)[0]
        normalization_factor = np.mean(data[..., b0_indices],axis=-1)
        data = data / np.repeat(normalization_factor[...,np.newaxis],np.shape(data)[-1],-1)
        if np.shape(data.shape)[0] == 1:
            njobs=1
        if data.shape[0] < njobs:
            njobs = 1
        if njobs > 1:
            # Flatten the indices first
            all_indices = list(np.ndindex(data.shape[:-1]))
            #np.save('data.npy', data)
            #data = np.load('data.npy', mmap_mode='r')
            # Define parallel function
            def parfun(ijk):
                single_voxel_data = np.array(data[ijk], copy=True)
                if not np.isnan(single_voxel_data[0]):
                    fit = self.ivim_fit(single_voxel_data, use_bvalues, **kwargs)
                    fit = self.D_and_Ds_swap(fit)
                else:
                    fit={'D':0,'f':0,'Dp':0}
                return ijk, fit


            # Run in parallel
            results_list = Parallel(n_jobs=njobs)(
                delayed(parfun)(ijk) for ijk in tqdm(all_indices, total=len(all_indices), mininterval=60) # updates every minute
            )

            # Initialize result arrays if not already done
            # Example: results = {key: np.zeros(data.shape[:-1]) for key in expected_keys}

            # Populate results after parallel loop
            for ijk, fit in results_list:
                for key in fit:
                    results[key][ijk] = fit[key]
        else:
            for ijk in tqdm(np.ndindex(data.shape[:-1]), total=np.prod(data.shape[:-1]), mininterval=60):
                # Normalize array
                single_voxel_data = data[ijk]
                if not np.isnan(single_voxel_data[0]):
                    args = [single_voxel_data, use_bvalues]
                    fit = self.D_and_Ds_swap(self.ivim_fit(*args, **kwargs)) # For single voxel fits, we assume this is a dict with a float value per key.
                else:
                    fit={'D':0,'f':0,'Dp':0}
                for key in list(fit.keys()):
                    results[key][ijk] = fit[key]
        #self.parameter_estimates = self.ivim_fit(data, bvalues)
        return results


    def osipi_fit_full_volume(self, data, bvalues=None, **kwargs):
        """
        Fit an entire volume of multi-b-value diffusion MRI data in a single call using the IVIM model.

        Unlike `osipi_fit`, which processes one voxel at a time, this method sends the full volume
        to the fitting algorithm. This can be more efficient for algorithms that support full-volume fitting.

        Parameters
        ----------
        data : np.ndarray
            2D (data x b-values), 3D (single slice), or 4D (multi-slice) diffusion-weighted imaging (DWI) data.
            The last dimension must correspond to the b-values.
        bvalues : array-like, optional
            Array of b-values corresponding to the last dimension of `data`. If not provided, the method
            uses `self.bvalues` set during object initialization.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to `ivim_fit_full_volume`.

        Returns
        -------
        results : dict of np.ndarray or bool
            Dictionary containing parametric maps for each IVIM parameter (keys from `self.result_keys`,
            or defaults ["f", "Dp", "D"]). Each value is an array matching the spatial dimensions of `data`.
            Returns `False` if full-volume fitting is not supported by the algorithm.

        Notes
        -----
        - This method does not normalize the input signal to the minimum b-value, unlike `osipi_fit`.

        Example
        -------
        # Standard usage:
        results = instance.osipi_fit_full_volume(data, bvalues=[0, 50, 200, 800])
        f_map = results['f']
        D_map = results['D']
        Dp_map = results['Dp']

        # If the algorithm does not support full-volume fitting:
        results = instance.osipi_fit_full_volume(data)
        if results is False:
            print("Full-volume fitting not supported.")
        """
        
        try:
            use_bvalues = bvalues if bvalues is not None else self.bvalues
            if use_bvalues is not None: use_bvalues = np.asarray(use_bvalues) 

            # Check if there is an attribute that defines the result dictionary keys
            if hasattr(self, "result_keys"):
                # result_keys is a list of strings of parameter names, e.g. "S0", "f1", "f2", etc.
                result_keys = self.result_keys
            else:
                # Default is ["f", "Dp", "D"]
                self.result_keys = ["f", "Dp", "D"]

            # Create the results dictionary
            results = {}
            for key in self.result_keys:
                results[key] = np.empty(list(data.shape[:-1]))
            # no normalisation as volume algorithms may not want normalized signals...
            args = [data, use_bvalues]
            fit = self.ivim_fit_full_volume(*args, **kwargs) # Assume this is a dict with an array per key representing the parametric maps
            for key in list(fit.keys()):
                results[key] = fit[key]

            return results

        except: 
            # Check if the problem is that full volume fitting is simply not supported in the standardized implementation
            if not hasattr(self, "ivim_fit_full_volume"): #and callable(getattr(self, "ivim_fit_full_volume")):
                print("Full volume fitting not supported for this algorithm")

            return False


    def osipi_print_requirements(self):
        """
        Prints the requirements of the algorithm.
        Attributes that are currently checked for are:
        required_bvalues (int):
            The lowest number of b-values required.
        required_thresholds (array-like)
            1D array-like of two ints [least number of b-value thresholds, most number of b-value_thresholds].
        required_bounds (bool):
            Whether bounds are required or not.
        required_bounds_optional (bool):
            Whether bounds are optional or not
        required_initial_guess (bool):
            Whether an initial guess is required or not.
        required_initial_guess_optional (bool):
            Whether an initial guess is optional or not.
        """
        print("\n### Algorithm requirements ###")
        
        # Check if the attributes exist and print them
        if hasattr(self, "required_bvalues"):
            print(f"Number of b-values: {self.required_bvalues}")
            
        if hasattr(self, "required_thresholds"):
            print(f"Numer of b-value thresholds [at least, at most]: {self.required_thresholds}")
            
        if hasattr(self, "required_bounds") and hasattr(self, "required_bounds_optional"):
            if self.required_bounds:
                print(f"Bounds required: {self.required_bounds}")
            else:
                # Bounds may not be required but can be optional
                if self.required_bounds_optional:
                    print(f"Bounds required: {self.required_bounds} but is optional")
                else:
                    print(f"Bounds required: {self.required_bounds} and is not optional")
                    
        if hasattr(self, "required_initial_guess") and hasattr(self, "required_initial_guess_optional"):
            if self.required_initial_guess:
                print(f"Initial guess required: {self.required_initial_guess}")
            else:
                # Bounds may not be required but can be optional
                if self.required_initial_guess_optional:
                    print(f"Initial guess required: {self.required_initial_guess} but is optional")
                else:
                    print(f"Initial guess required: {self.required_initial_guess} and is not optional")

    def osipi_accepted_dimensions(self):
        """The array of accepted dimensions
        e.g.
        (1D,   2D,   3D,    4D,    5D,    6D)
        (True, True, False, False, False, False)
        """
        
        #return (False,) * 6
        return True

    def osipi_accepts_dimension(self, dim):
        """Query if the selection dimension is fittable"""
        
        #accepted = self.accepted_dimensions()
        #if dim < 0 or dim > len(accepted):
            #return False
        #return accepted[dim]
        return True
    
    def osipi_check_required_bvalues(self):
        """Checks if the input bvalues fulfil the algorithm requirements"""
        
        #if self.bvalues.size < self.required_bvalues:
            #print("Conformance error: Number of b-values.")
            #return False
        #else: 
            #return True
        return True
        
    def osipi_check_required_thresholds(self):
        """Checks if the number of input thresholds fulfil the algorithm requirements"""
        
        #if (len(self.thresholds) < self.required_thresholds[0]) or (len(self.thresholds) > self.required_thresholds[1]):
            #print("Conformance error: Number of thresholds.")
            #return False
        #else: 
            #return True
        return True
        
    def osipi_check_required_bounds(self):
        """Checks if input bounds fulfil the algorithm requirements"""
        #if self.required_bounds is True and self.bounds is None:
            #print("Conformance error: Bounds.")
            #return False
        #else:
            #return True
        return True

    def osipi_check_required_initial_guess(self):
        """Checks if input initial guess fulfil the algorithm requirements"""
        
        #if self.required_initial_guess is True and self.initial_guess is None:
            #print("Conformance error: Initial guess")
            #return False
        #else:
            #return True
        return True

    
    def osipi_check_required_bvalues(self):
        """Minimum number of b-values required"""
        pass

    def osipi_author(self):
        """Author identification"""
        return ''
    
    def osipi_simple_bias_and_RMSE_test(self, SNR, bvalues, f, Dstar, D, noise_realizations=100):
        # Generate signal
        bvalues = np.asarray(bvalues)
        signals = f*np.exp(-bvalues*Dstar) + (1-f)*np.exp(-bvalues*D)
        
        f_estimates = np.zeros(noise_realizations)
        Dstar_estimates = np.zeros(noise_realizations)
        D_estimates = np.zeros(noise_realizations)
        for i in range(noise_realizations):
            # Add some noise
            sigma = signals[0]/SNR
            noised_signal = np.array([norm.rvs(signal, sigma) for signal in signals])
            
            # Perform fit with the noised signal
            # f_estimates[i], Dstar_estimates[i], D_estimates[i] = self.D_and_Ds_swap(self.ivim_fit(noised_signal, bvalues))
            result = self.ivim_fit(noised_signal, bvalues)
            f_estimates[i], Dstar_estimates[i], D_estimates[i] = result['f'], result['Dp'], result['D']
            
        # Calculate bias
        f_bias = np.mean(f_estimates) - f
        Dstar_bias = np.mean(Dstar_estimates) - Dstar
        D_bias = np.mean(D_estimates) - D
            
        # Calculate RMSE
        f_RMSE = np.sqrt(np.var(f_estimates) + f_bias**2)
        Dstar_RMSE = np.sqrt(np.var(Dstar_estimates) + Dstar_bias**2)
        D_RMSE = np.sqrt(np.var(D_estimates) + D_bias**2)
            
        print(f"f bias:     {f_bias:.4g}    \nf RMSE:     {f_RMSE:.4g}")
        print(f"Dstar bias: {Dstar_bias:.4g}\nDstar RMSE: {Dstar_RMSE:.4g}")
        print(f"D bias:     {D_bias:.4g}    \nD RMSE:     {D_RMSE:.4g}")

    
    def D_and_Ds_swap(self,results):
        if results['D']>results['Dp']:
            D=results['Dp']
            results['Dp']=results['D']
            results['D']=D
            results['f']=1-results['f']
        return results