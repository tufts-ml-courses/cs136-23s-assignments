import numpy as np

def softplus(x):
    """ Numerically stable transform from real line to positive reals

    Returns np.log(1.0 + np.exp(x))

    Fully vectorized and numerically stable for any input x

    Args
    ----
    x : array of values in (-\infty, +\infty)

    Returns
    -------
    ans : array of values in (0, +\infty), same size as x

    Examples
    --------
    >>> np.set_printoptions(precision=5)
    >>> softplus(5000.)
    5000.0
    >>> softplus(np.asarray([1.0]))
    array([1.31326])
    >>> softplus(np.asarray([0.0]))
    array([0.69315])
    >>> softplus(np.asarray([-1.0]))
    array([0.31326])
    >>> softplus(np.asarray([-5000.0]))
    array([0.])

    >>> np.set_printoptions(precision=5, suppress=True)
    >>> softplus(np.asarray([-100.0, -20.0, -2.0, 0.0, 2.0, 20.0]))
    array([ 0.     ,  0.     ,  0.12693,  0.69315,  2.12693, 20.     ])
    """
    if isinstance(x, float):
        if x < 5:
            # Standard definition
            return np.log1p(np.exp(x))
        else:
            # Numerically safe, avoids exp(x) overflow
            return x + np.log1p(np.exp(-x))
    # Vectorized version
    x_N = np.asarray(x)
    mask1_N = x > 5
    mask0_N = np.logical_not(mask1_N)
    out_N = np.zeros(x_N.shape, dtype=np.float64)
    out_N[mask0_N] = np.log1p(np.exp(x_N[mask0_N]))
    out_N[mask1_N] = x_N[mask1_N] + np.log1p(np.exp(-x_N[mask1_N]))
    return out_N
