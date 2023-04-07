'''
Define invertible, differentiable transform for arrays of positive vals.

Notes
=====

Define the "common" (aka constrained) parameter:
p : 1D array, size F, of positive real values

Define the unconstrained parameter:
x: 1D array, size F, of unconstrained real values

Transform from unconstrained to common: SOFTPLUS
p = log(exp(x) + 1)
Function `to_common_arr` does this in numerically stable way.

Transform from common to unconstrained: INVERSE of SOFTPLUS
x = log(exp(p) - 1)
Function `to_unconstrained_arr` does this in numerically stable way.

Verification
============
To run automated tests to verify correctness of this script, do:
(Add -v flag for verbose output)
```
$ python -m doctest transform__all_positive_entries.py
```

Examples
========
>>> ag_np.set_printoptions(suppress=False, precision=3, linewidth=80)

# Create 1D array of pos values ranging from 0.000001 to 1000000 
>>> p_F = ag_np.asarray([1e-6, 1e-3, 1e0, 1e3, 1e6])
>>> print(p_F)
[1.e-06 1.e-03 1.e+00 1.e+03 1.e+06]

# Look at transformed values, which vary from -13 to 1000000
>>> x_F = to_unconstrained_arr(p_F)
>>> print(x_F)
[-1.382e+01 -6.907e+00  5.413e-01  1.000e+03  1.000e+06]

# Show that the transform is invertible
>>> print(to_common_arr(to_unconstrained_arr(p_F)))
[1.e-06 1.e-03 1.e+00 1.e+03 1.e+06]
>>> ag_np.allclose(p_F, to_common_arr(to_unconstrained_arr(p_F)))
True

# Show that the transform is differentiable
>>> f = lambda v: ag_np.sum(ag_np.square(to_common_arr(v)))
>>> g = autograd.grad(f)
>>> g(to_unconstrained_arr(p_F))
array([2.000e-12, 1.999e-06, 1.264e+00, 2.000e+03, 2.000e+06])
'''

import autograd
import autograd.numpy as ag_np
from autograd.extend import primitive, defvjp

@primitive
def to_common_arr(x):
    """ Numerically stable transform from real line to positive reals

    Returns ag_np.log(1.0 + ag_np.exp(x))

    Autograd friendly and fully vectorized

    Args
    ----
    x : array of values in (-\infty, +\infty)

    Returns
    -------
    ans : array of values in (0, +\infty), same size as x
    """
    if not isinstance(x, float):
        mask1 = x > 0
        mask0 = ag_np.logical_not(mask1)
        out = ag_np.zeros_like(x)
        out[mask0] = ag_np.log1p(ag_np.exp(x[mask0]))
        out[mask1] = x[mask1] + ag_np.log1p(ag_np.exp(-x[mask1]))
        return out
    if x > 0:
        return x + ag_np.log1p(ag_np.exp(-x))
    else:
        return ag_np.log1p(ag_np.exp(x))

def make_grad__to_common_arr(ans, x):
    x = ag_np.asarray(x)
    def gradient_product(g):
        return ag_np.full(x.shape, g) * ag_np.exp(x - ans)
    return gradient_product

defvjp(to_common_arr, make_grad__to_common_arr)


@primitive
def to_unconstrained_arr(p):
    """ Numerically stable transform from positive reals to real line

    Implements ag_np.log(ag_np.exp(x) - 1.0)

    Autograd friendly and fully vectorized

    Args
    ----
    p : array of values in (0, +\infty)

    Returns
    -------
    ans : array of values in (-\infty, +\infty), same size as p
    """
    ## Handle numpy array case
    if not isinstance(p, float):
        mask1 = p > 10.0
        mask0 = ag_np.logical_not(mask1)
        out = ag_np.zeros_like(p)
        out[mask0] =  ag_np.log(ag_np.expm1(p[mask0]))
        out[mask1] = p[mask1] + ag_np.log1p(-ag_np.exp(-p[mask1]))
        return out
    ## Handle scalar float case
    else:
        if p > 10:
            return p + ag_np.log1p(-ag_np.exp(-p))
        else:
            return ag_np.log(ag_np.expm1(p))

def make_grad__to_unconstrained_arr(ans, x):
    x = ag_np.asarray(x)
    def gradient_product(g):
        return ag_np.full(x.shape, g) * ag_np.exp(x - ans)
    return gradient_product

defvjp(to_unconstrained_arr, make_grad__to_unconstrained_arr)
