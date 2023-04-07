'''
calc_penalty_stddev.py

Defines a penalty function that avoids pathological tendency of GMM to have
one component favor zero variance and infinite likelihood.

Penalty function is designed to be auto-differentiable via autograd package.

Usage
-----
Import in other modules to access `calc_penalty_stddev` function

To visualize the penalty function, execute this script to make a plot

$ python calc_penalty_stddev.py

'''
try:
    import autograd.numpy as my_np
except ImportError:
    import numpy as my_np

import numpy as np
import matplotlib.pyplot as plt

def calc_penalty_stddev(stddev_KD,
        variance_penalty_mode,
        variance_penalty_spread):
    ''' Calculate penalty on standard deviation parameter

    Computes penalty at each element of provided stddev array separately.
    Returns the sum of the penalties across all elements.

    Args
    ----
    stddev_KD : 2D array, shape (K, D)
        Stddev parameter for each cluster, feature dimension.
        Every entry must be positive.

    Returns
    -------
    penalty : float
        Penalty function evaluated at provided sigma array.
        If stddev_KD is an array, we sum across all dimensions.
    '''
    m = variance_penalty_mode
    s = variance_penalty_spread
    coef_A = 1.0/(s * m * m)
    coef_B = 1.0/(s * m)
    return (
        my_np.sum(coef_A * my_np.log(stddev_KD) 
        + 0.5 * coef_B / my_np.square(stddev_KD))
        )


if __name__ == '__main__':
    fig, ax_grid = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12,2.5))

    mode_list = [0.5, 1.0, 3.0]
    spread_list = [1.0, 5.0, 25.0]

    # Set up grid of G possible stddev we will evaluate
    G = 1001
    stddev_G = np.logspace(-2, 2, G) # from 0.01 to 100

    for mm, mode in enumerate(mode_list):
        for spread in spread_list:
            ax_grid[mm].plot(
                np.square(stddev_G),
                [calc_penalty_stddev(stddev, mode, spread) for stddev in stddev_G],
                marker='.',
                markersize=5.0,
                linestyle='-',
                label="spread = %.1f" % (spread))
        ax_grid[mm].legend(loc='upper right')
        ax_grid[mm].set_xlabel("$\sigma^2$")
        ax_grid[mm].set_title("mode = %.1f" % mode)
        ax_grid[mm].set_xlim([0, 6.0])
        ax_grid[mm].set_ylim([-0.2, 1.5])
        if mm == 0:
            ax_grid[mm].set_ylabel("penalty($\sigma$)")

    plt.tight_layout()
    plt.show()
