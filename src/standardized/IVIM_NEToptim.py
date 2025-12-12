from src.wrappers.OsipiBase import OsipiBase
import numpy as np
import IVIMNET.deep as deep
import torch
import warnings
from utilities.data_simulation.GenerateData import GenerateData

class IVIM_NEToptim(OsipiBase):
    """
    Bi-exponential fitting algorithm by Oliver Gurney-Champion, Amsterdam UMC
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Oliver Gurney Champion, Amsterdam UMC"
    id_algorithm_type = "Deep learnt bi-exponential fit"
    id_return_parameters = "f, D*, D, S0"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"

    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0,
                           0]  # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True  # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = False
    accepted_dimensions = 1  # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?


    # Supported inputs in the standardized class
    supported_bounds = True
    supported_initial_guess = False
    supported_thresholds = False

    def __init__(self, SNR=None, bvalues=None, thresholds=None, bounds=None, initial_guess=None, fitS0=True, traindata=None, n=5000000):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        if bvalues is None:
            raise ValueError("for deep learning models, bvalues need defining at initiaition")
        #super(OGC_AmsterdamUMC_biexp, self).__init__(bvalues, bounds, initial_guess, fitS0)
        super(IVIM_NEToptim, self).__init__(bvalues=bvalues, bounds=bounds, initial_guess=initial_guess)
        self.fitS0=fitS0
        self.bvalues=np.array(bvalues)
        self.initialize(bounds, initial_guess, fitS0, traindata, SNR, n)

    def initialize(self, bounds, initial_guess, fitS0, traindata, SNR, n):
        self.fitS0=fitS0
        self.deep_learning = True
        self.supervised = False
        # Additional options
        self.stochastic = True

        if traindata is None:
            warnings.warn('no training data provided (traindata = None). Training data will be simulated')
            if SNR is None:
                warnings.warn('No SNR indicated. Data simulated with SNR = (5-100)')
                SNR = (5, 100)
            self.training_data(self.bvalues,n=n,SNR=SNR)
        self.arg=Arg()
        print('note that the bounds in the network are soft bounds and implemented by a sigmoid transform. In order for the network to be sensitive over the range, we extend the bounds ny 30%')
        if bounds is not None:
            self.bounds = bounds
        else:
            warnings.warn('No bounds indicated. default bounds are used of         self.cons_min = [0, 0, 0.005, 0]   and self.cons_max = [0.005, 1, 0.2, 2.0]  # Dt, Fp, Ds, S0')
            self.bounds = {"S0" : [0, 2], "f" : [0, 1], "Dp" : [0.005, 0.2], "D" : [0, 0.005]} # These are defined as [lower, upper]
        self.arg.net_pars.cons_min = np.array([self.bounds["D"][0], self.bounds["f"][0], self.bounds["Dp"][0], self.bounds["S0"][0]])#bounds[0]  # Dt, Fp, Ds, S0
        self.arg.net_pars.cons_max = np.array([self.bounds["D"][1], self.bounds["f"][1], self.bounds["Dp"][1], self.bounds["S0"][1]])#bounds[1]  # Dt, Fp, Ds, S0

        boundsrange = 0.3 * (np.array(self.arg.net_pars.cons_max)-np.array(self.arg.net_pars.cons_min)) # ensure that we are on the most lineair bit of the sigmoid function
        self.arg.net_pars.cons_min = np.array(self.arg.net_pars.cons_min) - boundsrange
        self.arg.net_pars.cons_max = np.array(self.arg.net_pars.cons_max) + boundsrange
        self.bounds={"S0" : [self.arg.net_pars.cons_min[3], self.arg.net_pars.cons_max[3]], "f" : [self.arg.net_pars.cons_min[1], self.arg.net_pars.cons_max[1]], "Dp" : [self.arg.net_pars.cons_min[2], self.arg.net_pars.cons_max[2]], "D" : [self.arg.net_pars.cons_min[0], self.arg.net_pars.cons_max[0]]} # These are defined as [lower, upper]

        self.use_bounds = {"f": True, "Dp": True, "D": True}
        self.use_initial_guess = {"f": False, "Dp": False, "D": False}
        if traindata is None:
            self.net = deep.learn_IVIM(self.train_data['data'], self.bvalues, self.arg)
        else:
            self.net = deep.learn_IVIM(traindata, self.bvalues, self.arg)
        self.algorithm =lambda data: deep.predict_IVIM(data, self.bvalues, self.net, self.arg)


    def ivim_fit(self, signals, bvalues, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """
        if not np.array_equal(bvalues, self.bvalues):
            raise ValueError("bvalue list at fitting must be identical as the one at initiation, otherwise it will not run")

        paramsNN = deep.predict_IVIM(signals, self.bvalues, self.net, self.arg)

        results = {}
        results["D"] = paramsNN[0]
        results["f"] = paramsNN[1]
        results["Dp"] = paramsNN[2]

        return results


    def ivim_fit_full_volume(self, signals, bvalues, retrain_on_input_data=False, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """
        if not np.array_equal(bvalues, self.bvalues):
            raise ValueError("bvalue list at fitting must be identical as the one at initiation, otherwise it will not run")

        signals, shape = self.reshape_to_voxelwise(signals)
        if retrain_on_input_data:
            self.net = deep.learn_IVIM(signals, self.bvalues, self.arg, net=self.net)
        paramsNN = deep.predict_IVIM(signals, self.bvalues, self.net, self.arg)

        results = {}
        results["D"] = np.reshape(paramsNN[0],shape[:-1])
        results["f"] = np.reshape(paramsNN[1],shape[:-1])
        results["Dp"] = np.reshape(paramsNN[2],shape[:-1])

        return results

    def reshape_to_voxelwise(self, data):
        """
        reshapes multi-D input (spatial dims, bvvalue) data to 2D voxel-wise array
        Args:
            data (array): mulit-D array (data x b-values)
        Returns:
            out (array): 2D array (voxel x b-value)
        """
        B = data.shape[-1]
        voxels = int(np.prod(data.shape[:-1]))  # e.g., X*Y*Z
        return data.reshape(voxels, B), data.shape


    def training_data(self, bvalues, data=None, SNR=(5,100), n=5000000,Drange=(0.0003,0.0035),frange=(0,1),Dprange=(0.005,0.12),rician_noise=False):
        rng = np.random.RandomState(42)
        if data is None:
            gen = GenerateData(rng=rng)
            data, D, f, Dp = gen.simulate_training_data(bvalues, SNR=SNR, n=n,Drange=Drange,frange=frange,Dprange=Dprange,rician_noise=rician_noise)
            if self.supervised:
                self.train_data = {'data':data,'D':D,'f':f,'Dp':Dp}
            else:
                self.train_data = {'data': data}

class NetArgs:
    def __init__(self):
        self.optim = 'adam'  # these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
        self.lr = 0.00003  # this is the learning rate.
        self.patience = 10  # this is the number of epochs without improvement that the network waits untill determining it found its optimum
        self.batch_size = 128  # number of datasets taken along per iteration
        self.maxit = 500  # max iterations per epoch
        self.split = 0.9  # split of test and validation data
        self.load_nn = False  # load the neural network instead of retraining
        self.loss_fun = 'rms'  # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
        self.skip_net = False  # skip the network training and evaluation
        self.scheduler = False  # as discussed in the article, LR is important. This approach allows to reduce the LR itteratively when there is no improvement throughout an 5 consecutive epochs
        # use GPU if available
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = False
        # the optimized network settings

class NetPars:
    def __init__(self):
        self.dropout = 0.1  # 0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
        self.batch_norm = True  # False/True turns on batch normalistion
        self.parallel = 'parallel'  # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
        self.con = 'sigmoid'  # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
        self.tri_exp = False
        #### only if sigmoid constraint is used!
        self.cons_min = [0, 0, 0.005, 0]  # Dt, Fp, Ds, S0
        self.cons_max = [0.005, 0.8, 0.2, 2.0]  # Dt, Fp, Ds, S0
        ####
        self.fitS0 = True  # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
        self.depth = 2  # number of layers
        self.width = 0  # new option that determines network width. Putting to 0 makes it as wide as the number of b-values
        boundsrange = 0.3 * (np.array(self.cons_max)-np.array(self.cons_min)) # ensure that we are on the most lineair bit of the sigmoid function
        self.cons_min = np.array(self.cons_min) - boundsrange
        self.cons_max = np.array(self.cons_max) + boundsrange
class Arg:
    def __init__(self):
        self.train_pars = NetArgs()
        self.net_pars = NetPars()
