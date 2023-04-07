'''
Differentiable transform for parameters that are non-negative and sum to one.

Common parameter: log_pi_K

Unconstrained parameter: rho_K

Transform from unconstrained to constrained:
log_pi_K = rho_K - logsumexp(rho_K)

Examples
========
>>> import numpy as np
>>> np.set_printoptions(precision=3, suppress=1)

## Peaked probability vector
>>> pi_K = np.asarray([0.998, 0.001, 0.001])
>>> log_pi_K = to_common_arr(np.log(pi_K))
>>> log_pi_K
array([-0.002, -6.908, -6.908])
>>> logsumexp(log_pi_K)
0.0
>>> np.exp(log_pi_K)
array([0.998, 0.001, 0.001])

## Uniform probability vector
>>> log_pi_K = to_common_arr(np.asarray([0.0, 0.0, 0.0]))
>>> logsumexp(log_pi_K)
0.0
>>> np.exp(log_pi_K)
array([0.333, 0.333, 0.333])
'''

try:
    from autograd.scipy.special import logsumexp
except ImportError:
    from scipy.special import logsumexp

def to_common_arr(rho_K):
    ''' Convert unconstrained topic weights to proper normalized topics

    Should handle any non-nan, non-inf input without numerical problems.

    Args
    ----
    rho_K : 1D array, size K

    Returns
    -------
    log_pi_K : 1D array, size K
        Elementwise log of pi_K (a probability vector)
        Guaranteed to satisfy logsumexp(log_pi_K) == 0.0
    '''
    log_pi_K = rho_K - logsumexp(rho_K)
    return log_pi_K

def to_unconstrained_arr(log_pi_K):
    ''' Transform common arr to unconstrained space.

    Args
    ----
    log_pi_K : 1D array, size K
        Elementwise log of pi_K (a probability vector)
        Guaranteed to satisfy logsumexp(log_pi_K) == 0.0

    Returns
    -------
    rho_K : 1D array, size K
        Unconstrained real values.
    '''
    rho_K = 1.0 * log_pi_K
    return rho_K

