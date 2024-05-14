# BayesEmuOpt
Bayesian emulator optimisation (please cite Rogers et al. 2020; https://ui.adsabs.harvard.edu/abs/2019JCAP...02..031R/abstract)


Instantiate OptimisationClass object:

    Class to contain Bayesian emulator optimisation computations.
    
        get_objective [function([n_params]) --> scalar] - this is usually the ln posterior. ln prior can be sensible
        
        get_emulator_error [function([n_params]) --> array([n_data])] - return emulator error vector
        
        param_limits [n_params, 2 (lower, upper)] - limits of prior volume
        
        inverse_data_covariance [n_data, n_data] - inverse of data covariance matrix


Make optimisation proposal using make_proposal():
    
    Make a proposal for the next optimisation point.
    
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
                        Newton method
