'''
Purpose
-------
Sample from a 2-dim. Normal distribution using a Random Walk Sampler.
This is the Metropolis MCMC algorithm with a Gaussian proposal with controllable stddev

Target distribution:
# mean
>>> mu_D = np.asarray([-1.0, 1.0])
# covariance
>>> cov_DD = np.asarray([[2.0, 0.95], [0.95, 1.0]])

'''

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from RandomWalkSampler import RandomWalkSampler

def calc_target_log_pdf(z_D):
    ''' Compute log pdf of provided z value under target bivariate Normal distribution

    Args
    ----
    z_D : 1D array, size (D,)
        Value of the random variable at which we should compute log pdf

    Returns
    -------
    logpdf : float real scalar
        Log probability density function value at provided input
    '''
    # TODO compute logpdf of target distribution at z_D
    return -0.5 * np.sum(np.square(z_D)) # TODO FIXME

if __name__ == '__main__':
    n_burnin_samples = 5000   
    n_keep_samples   = 5000
    random_state = 42   # seed for random number generator
    prng = np.random.RandomState(random_state)

    # Two initializations, labeled 'A' and 'B'
    z_initA_D = np.asarray([-3.0,  3.0])
    z_initB_D = np.asarray([ 2.0, -2.0])

    # Look at a range of std. deviations
    rw_stddev_grid = np.asarray([0.01, 0.1, 1.0, 10.0])
    G = len(rw_stddev_grid)

    # Prepare a plot to view samples from two chains (A/B) side-by-side
    _, ax_grid = plt.subplots(
        nrows=2, ncols=G, sharex=True, sharey=True,
        figsize=(2*G, 2*2))

    for rr, rw_stddev in enumerate(rw_stddev_grid):

        # Create vector of size D of current proposal width
        rw_stddev_D = rw_stddev * np.ones(2)

        # TODO Create samplers and run them for specified num iterations
        # Make sure to provide rw_stddev_D and random_state as args

        # TODO Stack list of samples into a 2D array of size (S, D)
        # Remember, the samples in returned list *already discard* burnin
        zA_SD = prng.randn(n_keep_samples, 2)  # FIXME
        zB_SD = prng.randn(n_keep_samples, 2)  # FIXME

        # TODO unpack info about accept rates
        arA = 0.0 # FIXME use infoA returned by samplerA.draw_samples(...)
        arB = 0.0 # FIXME use infoB returned by samplerA.draw_samples(...)

        # Plot samples as scatterplot
        # Use small alpha transparency to visually debug rare/frequent samples
        ax_grid[0,rr].plot(zA_SD[:,0], zA_SD[:,1], 'r.', alpha=0.05)
        ax_grid[1,rr].plot(zB_SD[:,0], zB_SD[:,1], 'b.', alpha=0.05)
        # Mark initial points with "X"
        ax_grid[0,rr].plot(z_initA_D[0], z_initA_D[1], 'rx')
        ax_grid[1,rr].plot(z_initB_D[0], z_initB_D[1], 'bx')

        ax_grid[0,rr].text(-4, -3.5, 'Frac accept: % .3f' % arA)
        ax_grid[1,rr].text(-4, -3.5, 'Frac accept: % .3f' % arB)


    # Make plots pretty and standardized
    for ax in ax_grid.flatten():
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_aspect('equal', 'box')
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
    ax_grid[0,0].set_ylabel("z_1")
    ax_grid[1,0].set_ylabel("z_1")
    for col in range(G):
        ax_grid[0,col].set_title("rw_stddev = %.3f" % rw_stddev_grid[col])
        ax_grid[1,col].set_xlabel("z_0")

    plt.tight_layout()
    # TODO uncomment to save figure for report
    #plt.savefig("problem1_figure.png", bbox_inches='tight', pad_inches=0)
    #plt.savefig("problem1_figure.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()