'''
Summary
=======
Defines a ABSTRACT penalized ML estimator for Gaussian Mixture Models

This implementation provides an API for

* score
* get_params
* set_params

Concrete descendants need to implement 

* fit

in order to fulfill API of sklearn-like unsupervised learning models

'''

import numpy as np
from collections import defaultdict
import time
import pandas as pd

from calc_neg_log_lik__np import calc_neg_log_lik
from calc_penalty_stddev import calc_penalty_stddev


class GMM_PenalizedMLEstimator():
    """ Maximum Likelihood estimator for Gaussian mixture model

    Includes a penalty on std. dev. parameters to avoid pathology of some components
    with zero variance and infinite likelihood density

    Attributes
    ----------
    K : int
        Number of components
    D : int
        Number of data dimensions
    seed : int
        Seed for random number generator used for initialization
    variance_penalty_mode : float
        Must be positive.
        Defines mode of penalty on variance.
        See calc_penalty_stddev module.
    variance_penalty_spread : float,
        Must be positive.
        Defines spread of penalty on variance.
        See calc_penalty_stddev module.
    max_iter : int
        Maximum allowed number of iterations for training algorithm
    ftol : float
        Threshold that determines if training algorithm has converged
        Same definition as `ftol` setting used by scipy.optimize.minimize

    Additional Attributes (after calling fit)
    -----------------------------------------
    log_pi_K : 1D array, shape (K,)
        GMM parameter: Log of mixture weights
        Must satisfy logsumexp(log_pi_K) == 0.0 (which means sum(exp(log_pi_K)) == 1.0)
    mu_KD : 2D array, shape (K, D)
        GMM parameter: Means of all components
        The k-th row is the mean vector for the k-th component
    stddev_KD : 2D array, shape (K, D)
        GMM parameter: Standard Deviations of all components
        The k-th row is the stddev vector for the k-th component
    history : dict of lists
        Access performance metrics computed throughout iterative training.
        history['iter'] contains integer iteration count at each checkpoint
        history['train_loss_per_pixel'] contains training loss value at each checkpoint
        history['valid_score_per_pixel'] contains validation score at each checkpoint
            Normalized "per pixel" means divided by total number of observed feature dimensions (pixels)
            So that values for different size datasets can be fairly compared.
    """

    def __init__(self, K=1, D=1, seed=42,
            variance_penalty_mode=1.0, variance_penalty_spread=100.0,
            max_iter=1000, ftol=1e-9, do_double_check_correctness=False):
        ''' Constructor for GMM model estimator
    
        Args
        ----
        See class documentation for arguments, which define all attributes of this object.

        Returns
        -------
        New GMM_PenalizedMLEstimator_LBFGS object.
        ''' 
        self.K = int(K)
        self.D = int(D)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.ftol = float(ftol)
        self.do_double_check_correctness = bool(do_double_check_correctness)
        self.variance_penalty_spread = float(variance_penalty_spread)
        self.variance_penalty_mode = float(variance_penalty_mode)

    def get_params(self, deep=False):
        ''' Obtain key attributes for this object as a dictionary

        Needed for use with sklearn functionality (e.g. for selecting hyperparameters).

        Returns
        -------
        param_dict : dict
        '''
        return {
            'K': self.K,
            'D': self.D,
            'seed': self.seed,
            'max_iter': self.max_iter,
            'ftol': self.ftol,
            'variance_penalty_mode': self.variance_penalty_mode,
            'variance_penalty_spread': self.variance_penalty_spread,
            'do_double_check_correctness': self.do_double_check_correctness,
            }

    def set_params(self, **params_dict):
        ''' Set key attributes of this object from provided dictionary

        Needed for use with sklearn functionality (e.g. for selecting hyperparameters).

        Returns
        -------
        self. Internal attributes updated.
        '''
        for param_name, value in params_dict.items():
            setattr(self, param_name, value)
        return self

    def score(self, x_ND):
        ''' Compute log likelihood of provided dataset under this GMM

        Args
        ----
        x_ND : 2D array, shape (N, D)
            Dataset to evaluate likelihood
            Each row contains one feature vector of size D

        Returns
        -------
        log_lik : float
            log likelihood of dataset
        '''
        return -1.0 * calc_neg_log_lik(
            x_ND, self.log_pi_K, self.mu_KD, self.stddev_KD)

    def calc_penalty_stddev(self):
        ''' Compute penalty of the provided stddev parameters

        See calc_penalty_stddev.py for details of the penalty.

        Args
        ----
        None. Uses internal attribute self.stddev_KD (set by calling fit on this object) as input.

        Returns
        -------
        penalty : float
            Penalty function value at current stddev parameters for this GMM
        '''
        return calc_penalty_stddev(
            self.stddev_KD, self.variance_penalty_mode, self.variance_penalty_spread)

    def generate_initial_parameters(self, x_ND):
        ''' Generate initial GMM parameters using this object's random seed

        Tries to provide a sensible initialization of weights, means, and variances.
        Weights will be set to uniform probabilities = [1/K, 1/K, ... 1/K]
        Means will be set equal to a randomly chosen vector from training set.
            Each example has equal probability of being chosen.
        Variances will be set equal to a random value between (m, 2 * m)
            where m = variance_penalty_mode, the provided mode of the variance penalty
            The mode is intended to be a "reasonable" value for an empty component.
            Thus, choosing uniformly between (mode, 2*mode) is likely to be decent.
            We'd probably rather err too big than too small.

        Args
        ----
        x_ND : 2D array, shpae (N, D)
            Dataset used for training.
            Each row is a 'raw' feature vector of size D

        Returns
        -------
        log_pi_K : 1D array, shape (K,)
            GMM parameter: Log of mixture weights
            Must satisfy logsumexp(log_pi_K) == 0.0 (which means sum(exp(log_pi_K)) == 1.0)
        mu_KD : 2D array, shape (K, D)
            GMM parameter: Means of all components
            The k-th row is the mean vector for the k-th component
        stddev_KD : 2D array, shape (K, D)
            GMM parameter: Standard Deviations of all components
            The k-th row is the stddev vector for the k-th component
        '''
        assert self.D == x_ND.shape[1]

        prng = np.random.RandomState(self.seed)
        # Make every component equally probable
        log_pi_K = np.log(1.0 / self.K * np.ones(self.K))

        # Select each mean vector to be centered on a randomly chosen data point
        N = x_ND.shape[0]
        if N < self.K:
            x_ND = np.vstack([x_ND, np.random.randn(self.K, self.D)])
            N = x_ND.shape[0]
        chosen_rows_K = prng.choice(np.arange(N), self.K, replace=False)
        mu_KD = x_ND[chosen_rows_K].copy()

        # Select every variance initially as uniform between (mode, 2.0 * mode)
        # So that variances are not too small under prior knowledge
        stddev_KD = np.sqrt(self.variance_penalty_mode) * (
            1 + prng.rand(self.K, self.D))

        return log_pi_K, mu_KD, stddev_KD

    def write_history_to_csv(self, csv_path):
        ''' Write history of training to comma separated value (CSV) file

        Args
        ----
        csv_path : str
            String specifying location on local file system to save a new CSV file.

        Post Condition
        --------------
        CSV file created in provided filepath, with columns for each entry in the history dict.

        Returns
        -------
        None.
        '''
        df = pd.DataFrame()
        for key in self.history:
            cur_list = self.history[key]
            if df.shape[0] == 0 or df.shape[0] == len(cur_list):
                df[key] = cur_list
        df.to_csv(csv_path, index=False)
