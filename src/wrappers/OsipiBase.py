import numpy as np
import importlib
from scipy.stats import norm
import pathlib
import sys
from tqdm import tqdm

class OsipiBase:
    """The base class for OSIPI IVIM fitting"""
    
    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, algorithm=None, **kwargs):
        # Define the attributes as numpy arrays only if they are not None
        self.bvalues = np.asarray(bvalues) if bvalues is not None else None
        self.thresholds = np.asarray(thresholds) if thresholds is not None else None
        self.bounds = np.asarray(bounds) if bounds is not None else None
        self.initial_guess = np.asarray(initial_guess) if initial_guess is not None else None
        self.use_bounds = True
        self.use_initial_guess = True

        # Validate the inputs
        if self.bvalues is not None:
            self.osipi_check_required_bvalues()
        if self.thresholds is not None:
            self.osipi_check_required_thresholds()
        if self.bounds is not None:
            self.osipi_check_required_bounds()
        if self.initial_guess is not None:
            self.osipi_check_required_initial_guess()
        
        # If the user inputs an algorithm to OsipiBase, it is intereprete as initiating
        # an algorithm object with that name.
        if algorithm:
            self.osipi_initiate_algorithm(algorithm, bvalues=bvalues, thresholds=thresholds, bounds=bounds, initial_guess=initial_guess, **kwargs)
            
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
    def osipi_fit(self, data, bvalues=None, **kwargs):
        """Fits the data with the bvalues
        Returns [S0, f, Dstar, D]
        """
        self.osipi_validate_inputs()

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
        #results = np.empty(list(data.shape[:-1])+[3]) # Create an array with the voxel dimensions + the ones required for the fit
        #for ijk in tqdm(np.ndindex(data.shape[:-1]), total=np.prod(data.shape[:-1])):
            #args = [data[ijk], use_bvalues]
            #fit = list(self.ivim_fit(*args, **kwargs))
            #results[ijk] = fit

        for ijk in tqdm(np.ndindex(data.shape[:-1]), total=np.prod(data.shape[:-1])):
            args = [data[ijk], use_bvalues]
            fit = self.ivim_fit(*args, **kwargs) # For single voxel fits, we assume this is a dict with a float value per key.
            for key in list(fit.keys()):
                results[key][ijk] = fit[key]
        
        #self.parameter_estimates = self.ivim_fit(data, bvalues)
        return results
    
    def osipi_fit_full_volume(self, data, bvalues=None, **kwargs):
        """Sends a full volume in one go to the fitting algorithm. The osipi_fit method only sends one voxel at a time.

        Args:
            data (array): 3D (single slice) or 4D (multi slice) DWI data.
            bvalues (array, optional): The b-values of the DWI data. Defaults to None.

        Returns:
            results (dict): Dict with key each containing an array which is a parametric map.
        """

        self.osipi_validate_inputs()
        
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

    def osipi_validate_inputs(self):
        """Validates the inputs of the algorithm."""
        self.osipi_check_required_bvalues()
        self.osipi_check_required_thresholds()
        self.osipi_check_required_bounds()
        self.osipi_check_required_initial_guess()
        self.osipi_accepts_dimension(self.data.ndim)

    def osipi_accepted_dimensions(self):
        """The array of accepted dimensions
        e.g.
        (1D,   2D,   3D,    4D,    5D,    6D)
        (True, True, False, False, False, False)
        """
        
        return getattr(self, 'accepted_dimensions', (1, 3))

    def osipi_accepts_dimension(self, dim):
        """Query if the selection dimension is fittable"""
        
        min_dim, max_dim = self.osipi_accepted_dimensions()
        if not (min_dim <= dim <= max_dim):
            raise ValueError(f"Dimension {dim}D not supported. Requires {min_dim}-{max_dim}D spatial data")
        return True
    
    def osipi_check_required_bvalues(self):
        """Checks if the input bvalues fulfil the algorithm requirements"""
        if not hasattr(self, "required_bvalues"):
            raise AttributeError("required_bvalues not defined for this algorithm")
        if self.bvalues is None:
            raise ValueError("bvalues are not provided")
        if len(self.bvalues) < self.required_bvalues:
            raise ValueError(f"Atleast {self.required_bvalues} are required, but only {len(self.bvalues)} were provided")
        return True
        
    def osipi_check_required_thresholds(self):
        """Checks if the number of input thresholds fulfil the algorithm requirements"""
        if not hasattr(self, "required_thresholds"):
            raise AttributeError("required_thresholds is not defined for this algorithm")
        if self.thresholds is None:
            raise ValueError("thresholds are not provided")
        if len(self.thresholds) < self.required_thresholds[0] and len(self.thresholds) > self.required_thresholds[1]:
            raise ValueError(f"Thresholds should be between {self.required_thresholds[0]} and {self.required_thresholds[1]} but {len(self.thresholds)} were provided")
        return True
        
    def osipi_check_required_bounds(self):
        """Checks if input bounds fulfil the algorithm requirements"""
        if self.required_bounds is False and self.bounds is None:
            raise ValueError("bounds are required but not provided")
        return True

    def osipi_check_required_initial_guess(self):
        """Checks if input initial guess fulfil the algorithm requirements"""
        if self.required_initial_guess is False and self.initial_guess is None:
            raise ValueError("initial_guess are required but not provided")
        return True

    def osipi_author():
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
            f_estimates[i], Dstar_estimates[i], D_estimates[i] = self.ivim_fit(noised_signal, bvalues)
            
        # Calculate bias
        f_bias = np.mean(f_estimates) - f
        Dstar_bias = np.mean(Dstar_estimates) - Dstar
        D_bias = np.mean(D_estimates) - D
            
        # Calculate RMSE
        f_RMSE = np.sqrt(np.var(f_estimates) + f_bias**2)
        Dstar_RMSE = np.sqrt(np.var(Dstar_estimates) + Dstar_bias**2)
        D_RMSE = np.sqrt(np.var(D_estimates) + D_bias**2)
            
        print(f"f bias:\t{f_bias}\nf RMSE:\t{f_RMSE}")
        print(f"Dstar bias:\t{Dstar_bias}\nDstar RMSE:\t{Dstar_RMSE}")
        print(f"D bias:\t{D_bias}\nD RMSE:\t{D_RMSE}")
            
    
