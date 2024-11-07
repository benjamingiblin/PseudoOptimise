import numpy as np
import numpy.random as npr
import numpy.testing as npt
import scipy.optimize as spo
from scipy.stats import truncnorm
from scipy.stats import multivariate_normal

# The following lines suppress the warnings from optimisation
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
# (note: I make no promises on if this is a good idea,
# but it saves 30MB output files being full of:
# /home/bengib/anaconda3/lib/python3.11/site-packages/scipy/optimize/_numdiff.py:576:
# RuntimeWarning: invalid value encountered in subtract
#  df = fun(x) - f0
# Think these just come from having a funny lhd surface.
# also means output file has up-to-date info DURING run instead of all printed at end.)

def map_from_unit_cube(param_vec, param_limits):
    """
    Map a parameter vector from the unit cube to the original dimensions of the space.
    Arguments:
    param_vec - the vector of parameters to map. Should all be [0,1]
    param_limits - the maximal limits of the parameters to choose.
    #Credit to Simeon Bird
    """
    assert (np.size(param_vec),2) == np.shape(param_limits)
    assert np.all((param_vec >= 0)*(param_vec <= 1))
    assert np.all(param_limits[:,0] <= param_limits[:,1])
    new_params = param_limits[:,0] + param_vec*(param_limits[:,1] - param_limits[:,0])
    assert np.all(new_params <= param_limits[:,1])
    assert np.all(new_params >= param_limits[:,0])
    return new_params

def map_to_unit_cube(param_vec, param_limits):
    """
    Map a parameter vector to the unit cube from the original dimensions of the space.
    Arguments:
    param_vec - the vector of parameters to map.
    param_limits - the limits of the allowed parameters.
    Returns:
    vector of parameters, all in [0,1].
    #Credit to Simeon Bird
    """
    assert (np.size(param_vec),2) == np.shape(param_limits)
    assert np.all(param_vec-1e-16 <= param_limits[:,1])
    assert np.all(param_vec+1e-16 >= param_limits[:,0])
    ii = np.where(param_vec > param_limits[:,1])
    param_vec[ii] = param_limits[ii,1]
    ii = np.where(param_vec < param_limits[:,0])
    param_vec[ii] = param_limits[ii,0]
    assert np.all(param_limits[:,0] <= param_limits[:,1])
    new_params = (param_vec-param_limits[:,0])/(param_limits[:,1] - param_limits[:,0])
    assert np.all((new_params >= 0)*(new_params <= 1))
    return new_params

def map_to_unit_cube_list(param_vec_list, param_limits):
    """Map multiple parameter vectors to the unit cube"""
    return np.array([map_to_unit_cube(param_vec, param_limits) for param_vec in param_vec_list])

def map_from_unit_cube_list(param_vec_list, param_limits):
    """Map multiple parameter vectors back from the unit cube"""
    return np.array([map_from_unit_cube(param_vec, param_limits) for param_vec in param_vec_list])


class OptimisationClass:
    """Class to contain Bayesian emulator optimisation computations.
        get_objective [function([n_params]) --> scalar] - this is usually the ln posterior. ln prior can be sensible
        get_emulator_error [function([n_params]) --> array([n_data])] - return emulator error vector
        param_limits [n_params, 2 (lower, upper)] - limits of prior volume
        inverse_data_covariance [n_data, n_data] - inverse of data covariance matrix"""
    def __init__(self, get_objective, get_emulator_error, param_limits, inverse_data_covariance, mvn):
        self.get_objective = get_objective
        self.get_emulator_error = get_emulator_error
        self.param_limits = param_limits
        self.inverse_data_covariance = inverse_data_covariance
        self.mvn = mvn # multivariate normal used in exploitation

    def exploration_weight_GP_UCB(self, nu, delta=0.5):
        """Choose the exploration weight for the GP-UCB acquisition function."""
        assert nu >= 0.
        assert 0. < delta < 1.
        return np.sqrt(nu * 2. * np.log((np.pi ** 2.) / 3. / delta))

    def exploration_GP_UCB(self, params, nu, **kwargs):
        """Evaluate the exploration term of the GP-UCB acquisition function."""
        emulator_error = self.get_emulator_error(params)
        return self.exploration_weight_GP_UCB(nu, **kwargs) * np.dot(emulator_error,
                                                                     np.dot(self.inverse_data_covariance, emulator_error))

    # The likelihood & posterior used by exploit'n. The Prior is used by both explor'n & exploit'n
    def lnlike(self, p):
        return self.mvn.logpdf(p) - self.mvn.logpdf(self.mvn.mean)
        # 2nd term rm's normalis'n off Gauss. (peaks at 0). 

    def lnprior(self, p):
        # p in unitary space has values [0,1]                                                                                
        # just exclude outer 5%:                                                     
        if p.max() > 0.95 or p.min() < 0.05:
            return -np.inf
        return 0.

    def lnprob(self, p):
        lp = self.lnprior(p)
        return lp + self.lnlike(p) if np.isfinite(lp) else -np.inf


    def exploitation_GP_UCB(self, params):
        """Evaluate the exploitation term of the GP-UCB acquisition function."""
        return self.lnprob(params) #/ lnprob(mean_gauss) 

    def acquisition_GP_UCB(self, params, nu, **kwargs):
        """Evaluate the modified GP-UCB acquisition function."""
        return -1*(self.exploitation_GP_UCB(params) + self.exploration_GP_UCB(params, nu, **kwargs))
        # avoid evaluating explor'n if params outside prior (get NaN's in proposal).
        #exploit = self.exploitation_GP_UCB(params) # this is where the prior gets applied
        #return -np.inf if not np.isfinite(exploit) else exploit + self.exploration_GP_UCB(params, nu, **kwargs)

    def optimise_acquisition_function(self, params_start='default', nu=0.19, acquisition='GP_UCB', bounds='default',
                                      method='TNC', **kwargs):
        """Find parameter vector at maximum of acquisition function."""
        if acquisition == 'GP_UCB':
            acquisition = lambda params: self.acquisition_GP_UCB(params, nu, **kwargs)
        else:
            raise ValueError('Unsupported acquisition function.')

        if bounds == 'default':
            bounds = [(0.1, 0.9) for _ in range(self.param_limits.shape[0])]

        if params_start == 'default':
            params_start = np.ones(self.param_limits.shape[0]) * 0.5
        else:
            params_start = map_to_unit_cube(params_start, self.param_limits)

        #acquisition_max = spo.minimize(acquisition, params_start, method=method, bounds=bounds).x
        acquisition_max = spo.basinhopping(acquisition, params_start).x
        #acquisition_max = spo.shgo(acquisition, bounds=bounds).x #shgo failed to add even 1 node
        return acquisition_max

    def make_proposal(self, params_start='default', nu=0.19, std_dev=None, acquisition='GP_UCB', bounds='default',
                      method='TNC', **kwargs):
        """Make a proposal for the next optimisation point.
            params_start [n_params] - initialisation for optimisation of acquisition function. Maximum posterior is
                            usually a good place to start for quick convergence but we recommend varying this to ensure
                            parameter space is fully explored. Defaults to mid-point of prior volume
            nu - hyper-parameter that sets relative balance between exploitation and exploration. Set higher to increase
                    exploration. We recommend varying this to get a sensible balance
            std_dev [n_params] - standard deviation for random exploration displacement. We recommend setting this to
                                    previous estimate for 1D marginalised 1 sigma constraints
            acquisition - only GP-UCB implemented at the moment
            bounds [n_params, tuple(lower, upper)] - in unit hypercube. Bounds for acquisition function maximisation.
                                                    Defaults to excluding outer 5% to account for Gaussian process error
                                                    exploding outside training set
            method - method for optimisation of acquisition function. See scipy.optimize.minimize. Defaults to truncated
                        Newton method"""
        
        acquisition_max = self.optimise_acquisition_function(params_start=params_start, nu=nu, acquisition=acquisition,
                                                             bounds=bounds, method=method, **kwargs)
        # what's the balance of exploit'n & explor'n at the proposal point? Store this.
        # the -1 here mimics the -1 which is implemented in the acquisition func above
        # makes it so acq = -sigma_GP*Sigma_data*sigma_GP + chi^2
        exploit = -1*self.exploitation_GP_UCB(acquisition_max) 
        explor = -1*self.exploration_GP_UCB(acquisition_max, nu)
        
        if std_dev is None:
            displacement = np.zeros(len(acquisition_max))
        else:
            # make it so the random Gauss is truncated so proposal can never exceed [0,1]:
            displacement = truncnorm.rvs(a = -1*acquisition_max, b = 1-acquisition_max,
                                         loc = 0, scale = std_dev, size = len(acquisition_max))
        proposal = map_from_unit_cube(acquisition_max + displacement, self.param_limits) 
        npt.assert_array_less(proposal, self.param_limits[:, 1],
                              err_msg='Proposal greater than parameter limits -- try again!')
        npt.assert_array_less(self.param_limits[:, 0], proposal,
                              err_msg='Proposal less than parameter limits -- try again!')

        return proposal, displacement, exploit, explor
