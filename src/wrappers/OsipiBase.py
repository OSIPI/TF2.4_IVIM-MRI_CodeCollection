import numpy as np
import importlib
from scipy.stats import norm
import pathlib
import sys

class OsipiBase:
    """The base class for OSIPI IVIM fitting"""
    
    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, algorithm=None, **kwargs):
        # Define the attributes as numpy arrays only if they are not None
        self.bvalues = np.asarray(bvalues) if bvalues is not None else None
        self.thresholds = np.asarray(thresholds) if thresholds is not None else None
        self.bounds = np.asarray(bounds) if bounds is not None else None
        self.initial_guess = np.asarray(initial_guess) if initial_guess is not None else None
        
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
    def osipi_fit(self, data, bvalues, **kwargs):
        """Fits the data with the bvalues
        Returns [S0, f, Dstar, D]
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
        args = [data, use_bvalues]
        #if self.required_bounds or self.required_bounds_optional:
            #args.append(use_bounds)
        #if self.required_initial_guess or self.required_initial_guess_optional:
            #args.append(use_initial_guess)
            
        # Run a check_requirements method that makes sure that these inputs fulfil the requirements
        
        
        # Then we check which inputs are required for the algorithm using the requirements attributes in the template 
        
        # Then we pass everything into the ivim_fit method which performs the fit according to the algorithm
        
        #args = [data, use_bvalues, use_initial_guess, use_bounds, use_thresholds]
        #args = [arg for arg in args if arg is not None]
        results = self.ivim_fit(*args, **kwargs)
        
        #self.parameter_estimates = self.ivim_fit(data, bvalues)
        return results
    
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

    
    def osipi_check_required_bvalues():
        """Minimum number of b-values required"""
        pass

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
            
    
