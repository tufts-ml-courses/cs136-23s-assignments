'''
Summary
=======
Defines a penalized ML estimator for Gaussian Mixture Models, using LBFGS gradient descent.

L-BFGS = limited-memory BFGS optimization

Provides a CONCRETE implementation of an sk-learn-like estimator API

* fit        : implemented here, via L-BFGS
* score
* get_params
* set_params

Examples
========
>>> np.set_printoptions(suppress=False, precision=3, linewidth=80)
>>> D = 2

## TEST 1: check on "empty" dataset with no observations

## 1a. Verify resulting variance is close to specified mode of 2.0 (within 0.02)

>>> gmm = GMM_PenalizedMLEstimator_LBFGS(K=3, D=2, seed=42, variance_penalty_mode=2.0)
>>> empty_ND = np.zeros((0,D))
>>> gmm.fit(empty_ND, verbose=False)
>>> gmm.message
'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'

>>> np.allclose(np.square(gmm.stddev_KD), 2.0, atol=0.02)
True

## TEST 2: check on randomly generated dataset with 3 well-separated Gaussian blobs

>>> N = 25
>>> prng = np.random.RandomState(8675309)
>>> x1_ND = 0.1 * prng.randn(N, D) + np.asarray([[0, 0]])
>>> x2_ND = 0.1 * prng.randn(N, D) + np.asarray([[-1, 0]])
>>> x3_ND = 0.1 * prng.randn(N, D) + np.asarray([[0, +1]])
>>> x_ND = np.vstack([x1_ND, x2_ND, x3_ND])

# 2a. Verify initial parameters are reasonable

>>> gmm = GMM_PenalizedMLEstimator_LBFGS(K=3, D=2, seed=402)
>>> log_pi_K, mu_KD, stddev_KD = gmm.generate_initial_parameters(x_ND)
>>> print(log_pi_K)
[-1.099 -1.099 -1.099]
>>> print(mu_KD)
[[-0.999 -0.046]
 [ 0.02   0.092]
 [ 0.014  0.915]]
>>> print(stddev_KD)
[[1.326 1.362]
 [1.439 1.652]
 [1.441 1.291]]

# 2b. Verify the likelihood calculation
>>> orig_score = -1.0 * calc_neg_log_lik(x_ND, log_pi_K, mu_KD, stddev_KD)
>>> print("%.3f" % orig_score)
-205.444

# 2c. Verify that training finds good parameters

>>> gmm.fit(x_ND, verbose=False)
>>> gmm.message
'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
>>> np.exp(gmm.log_pi_K)
array([0.333, 0.333, 0.333])
>>> gmm.mu_KD
array([[-0.002,  1.01 ],
       [-0.007,  0.01 ],
       [-1.008,  0.009]])

# 2d. Verify the final parameters have improved log likelihood (original was +205)

>>> print("%.3f" % gmm.score(x_ND))
61.819

# 2e. Examine loss history, verify loss goes down and converges

>>> print(np.asarray(gmm.history['train_loss_per_pixel'][:5]))
[1.37  0.869 0.68  0.586 0.524]
>>> print(np.asarray(gmm.history['train_loss_per_pixel'][-5:]))
[-0.39 -0.39 -0.39 -0.39 -0.39]

# 2f. Verify that final gradient is small (close to zero)

>>> print(np.asarray(gmm.history['grad_norm'][-5:]))
[2.090e-05 2.766e-05 2.762e-05 1.875e-05 9.130e-06]
'''

import numpy as np
from collections import defaultdict
import time
import pandas as pd

import scipy.optimize

import autograd
import autograd.numpy as ag_np

import transform__all_positive_arr
import transform__probability_vector

from GMM_PenalizedMLEstimator import GMM_PenalizedMLEstimator
from calc_penalty_stddev import calc_penalty_stddev
from calc_neg_log_lik__ag import calc_neg_log_lik


def to_unconstrained_parameters(log_pi_K, mu_KD, stddev_KD):
    ''' Convert provided parameters to unconstrained parameters via 

    * transform__probability_vector for converting log_pi_K to reals
    * transform__all_positive_arr for converting stddev (whare are positive) to reals

    Args
    ----
    log_pi_K : 1D array, shape (K,)
        GMM parameter: Log of mixture weights
        Must satisfy logsumexp(log_pi_K) == 0.0 (which means sum(exp(log_pi_K)) == 1.0)
    mu_KD : 2D array, shape (K, D)
        GMM parameter: Means of all components
        The k-th row is the mean vector for the k-th component
    stddev_KD : 2D array, shape (K, D)
        GMM parameter: Standard Deviations of all components
        The k-th row is the stddev vector for the k-th component

    Returns
    -------
    rho_K : 1D array, shape (K,)
        Unconstrained array representing mixture weights.
    mu_KD : 2D array, shape (K, D)
        GMM parameter for means (already unconstrained).
    nu_KD : 2D array, shape (K, D)
        Unconstrained array representing standard deviation parameters.
    '''
    return (
        transform__probability_vector.to_unconstrained_arr(log_pi_K),
        mu_KD,
        transform__all_positive_arr.to_unconstrained_arr(stddev_KD),
        )

def to_common_parameters(rho_K, mu_KD, nu_KD):
    ''' Convert provided unconstrained parameters to the common (constrained) versions

    Args
    ----
    rho_K : 1D array, shape (K,)
        Unconstrained array representing mixture weights.
    mu_KD : 2D array, shape (K, D)
        GMM parameter for means (already unconstrained).
    nu_KD : 2D array, shape (K, D)
        Unconstrained array representing standard deviation parameters.

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
    return (
        transform__probability_vector.to_common_arr(rho_K),
        mu_KD,
        transform__all_positive_arr.to_common_arr(nu_KD),
        )

class GMM_PenalizedMLEstimator_LBFGS(GMM_PenalizedMLEstimator):
    """ Maximum Likelihood estimator for Gaussian mixture model, trained with LBFGS

    Includes a penalty on std. dev. parameters to avoid pathology of some components
    with zero variance and infinite likelihood proba. density.

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

    # __init__ inherited from base class

    # get_params inherited from base class


    def to_flat_array_of_unconstrained_parameters(self, log_pi_K, mu_KD, stddev_KD):
        ''' Convert common GMM parameters to a flat 1D array of unconstrained parameters

        Args
        ----
        log_pi_K : 1D array, shape (K,)
            GMM parameter: Log of mixture weights
            Must satisfy logsumexp(log_pi_K) == 0.0 (which means sum(exp(log_pi_K)) == 1.0)
        mu_KD : 2D array, shape (K, D)
            GMM parameter: Means of all components
            The k-th row is the mean vector for the k-th component
        stddev_KD : 2D array, shape (K, D)
            GMM parameter: Standard Deviations of all components
            The k-th row is the stddev vector for the k-th component

        Returns
        -------
        vec_M : 1D array, shape (M,), where M = K + K*D + K*D
            Represents unconstrained parameters all packed into one vector.
        
        '''
        rho_K, mu_KD, s_KD = to_unconstrained_parameters(log_pi_K, mu_KD, stddev_KD)
        return ag_np.hstack([
            ag_np.reshape(rho_K, (self.K,)),
            ag_np.reshape(mu_KD, (self.K * self.D,)),
            ag_np.reshape(s_KD, (self.K * self.D,)),                
            ])

    def to_common_parameters_from_flat_array(self, vec_M):
        ''' Convert flat array of unconstrained parameters to the common GMM parameters

        Args
        ----
        vec_M : 1D array, shape (M,), where M = K + K*D + K*D
            Represents unconstrained parameters all packed into one vector.

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
        rho_K = vec_M[:self.K]
        mu_start = self.K
        mu_stop = mu_start + self.K * self.D
        mu_KD = ag_np.reshape(vec_M[mu_start:mu_stop], (self.K, self.D))

        s_start = mu_stop
        s_stop = s_start + self.K * self.D
        s_KD = ag_np.reshape(vec_M[s_start:s_stop], (self.K, self.D))
        return to_common_parameters(rho_K, mu_KD, s_KD)

    def fit(self, x_ND, x_valid_ND=None, verbose=True):
        ''' Fit this estimator to provided training data using LBFGS algorithm

        Args
        ----
        x_ND : 2D array, shape (N, D)
            Dataset used for training.
            Each row is an observed feature vector of size D
        x_valid_ND : 2D array, shape (Nvalid, D), optional
            Optional, dataset used for heldout validation.
            Each row is an observed feature vector of size D
            If provided, used to measure heldout likelihood at every checkpoint.
            These likelihoods will be recorded in self.history['valid_score_per_pixel']
        verbose : boolean, optional, defaults to True
            If provided, a message will be printed to stdout after every iteration,
            indicating the current training loss and (if possible) validation score.

        Returns
        -------
        self : this GMM object
            Internal attributes log_pi_K, mu_KD, stddev_KD updated.
            Performance metrics stored after every iteration in history 
        '''
        N = np.maximum(x_ND.shape[0], 1.0)

        ## Create history attribute to store progress at every checkpoint (every iteration)
        self.history = defaultdict(list)

        ## Create initial parameters at random, using self.seed for the random seed
        # Will always create same parameters if self.seed is the same value.
        log_pi_K, mu_KD, stddev_KD = self.generate_initial_parameters(x_ND)

        ## Package up parameters into one vector of unconstrained parameters
        init_param_vec = self.to_flat_array_of_unconstrained_parameters(log_pi_K, mu_KD, stddev_KD)

        ## Define loss fuction in terms of single vector containing all unconstrained parameters
        # Will compute the "per pixel" or "per dimension" loss
        def calc_loss(vec_M):
            ''' Compute per-pixel loss (negative log likelihood plus penalty)

            Returns
            -------
            loss : float
            '''
            # First, take current unconstrained parameters and transform back to common parameters
            # This provided transformation is autograd-able.
            log_pi_K, mu_KD, stddev_KD = self.to_common_parameters_from_flat_array(vec_M)

            # Second compute the loss
            # TODO replace this placeholder!
            loss_placeholder = ag_np.sum(ag_np.square(vec_M))

            # Finally, be sure this is per-pixel loss (total num pixels = N * D)
            return loss_placeholder / (N * self.D)

        ## Define gradient in terms of single vector of unconstrained parameters
        calc_grad = autograd.grad(calc_loss)
        calc_loss_and_grad = autograd.value_and_grad(calc_loss)

        ## Define callback function for monitoring progress of gradient descent
        # Will be called at every checkpoint (after every iteration of LBFGS)
        self.callback_count = 0
        self.start_time_sec = time.time()

        def callback_update_history(cur_param_vec):
            cur_loss, cur_grad_vec = calc_loss_and_grad(cur_param_vec)
            self.history['iter'].append(self.callback_count)
            self.history['train_loss_per_pixel'].append(cur_loss)

            log_pi_K, mu_KD, stddev_KD = self.to_common_parameters_from_flat_array(cur_param_vec)
            if x_valid_ND is None:
                valid_neg_log_lik_msg = "" # empty message when no validation set provided
            else:
                # TODO compute the per-pixel negative log likelihood on validation set
                va_score_per_pixel = 0.0123 + self.seed / 10000.0 # FIXME
                valid_score_msg = "| valid score % 9.6f" % (va_score_per_pixel)
                self.history['valid_score_per_pixel'].append(va_score_per_pixel)
            if verbose:
                print("iter %4d / %4d after %9.1f sec | train loss % 9.6f %s" % (
                    self.callback_count, self.max_iter,
                    time.time() - self.start_time_sec,
                    cur_loss,
                    valid_score_msg))

            self.history['grad_norm'].append(np.sum(np.abs(cur_grad_vec)) / cur_grad_vec.size)
            self.callback_count += 1

        ## Perform callback on initial parameters
        # Always good to know performance at original initialization
        callback_update_history(init_param_vec)

        ## Call LBFGS routine from scipy
        # This will perform many LBFGS update iterations,
        # and after each one will perform a callback using our provided function.
        # See scipy.optimize.minimize docs for details
        result = scipy.optimize.minimize(
            calc_loss,
            init_param_vec,
            jac=calc_grad,
            method='l-bfgs-b',
            constraints={},
            callback=callback_update_history,
            options=dict(maxiter=self.max_iter, ftol=self.ftol))

        ## Unpack the result of the optimization
        self.result = result
        self.message = str(result.message)
        optimal_param_vec = result.x
        self.log_pi_K, self.mu_KD, self.stddev_KD = \
            self.to_common_parameters_from_flat_array(optimal_param_vec)

