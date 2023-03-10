import numpy as np
import scipy.stats
import copy

class RandomWalkSampler(object):
    ''' Sampler for performing Metropolis MCMC with Random Walk proposals

    Requires a target random variable with continuous real sample space.
    
    Example Usage
    -------------
    # Define a target distribution (2-dim. standard normal)
    >>> def calc_tilde_log_pdf(z_D):
    ...     logpdf1 = scipy.stats.norm.logpdf(z_D[0], 0, 1)
    ...     logpdf2 = scipy.stats.norm.logpdf(z_D[1], 0, 1)
    ...     return logpdf1 + logpdf2

    # Create sampler
    >>> rw_stddev_D = .5 * np.ones(2)
    >>> sampler = RandomWalkSampler(
    ...     calc_tilde_log_pdf, [rw_stddev_D], random_state=42)

    # Draw samples starting a specified initial value
    >>> z_D_list, info = sampler.draw_samples(np.zeros(2), 30000)

    # Use samples in to estimate mean of the distribution
    >>> np.mean(np.vstack(z_D_list), axis=0)
    array([-0.00954532,  0.01338581])

    >>> info['accept_rate']
    0.7553

    Attributes
    ----------
    D : dimension
    calc_target_logpdf : function
        Given a value of random variable, computes model's posterior logpdf
        up to an additive constant
        Args:
        * z_D, 1D array
        Returns:
        * logpdf : scalar float
            Posterior's log probability density at provided value of array z_D
            Can be accurate up to an additive constant
    list_of_rw_stddev_D : list where each entry is a 1D array of size D
        Each entry represents a standard dev. vector for random walk proposal
        Can provide just one entry, or multiple entries.
        Each sample will generate via a sequence of
        Metropolis propose->decide transitions, one for each entry in list
    random_state : numpy.random.RandomState
        Pseudorandom number generator, supports .rand() and .randn()
    '''

    def __init__(self, calc_tilde_log_pdf, 
            list_of_rw_stddev_D=None,
            random_state=0):
        ''' Constructor for RandomWalkSampler object

        Args
        ----
        calc_tilde_log_pdf : function
            Given a value of random variable, computes the logpdf
            Args:
            * z_D, 1D array of size D
            Returns:
            * logpdf : scalar float
                Log probability density of provided sample
                Can be accurate up to an additive constant
        list_of_rw_stddev_D : list where each entry is a 1D array of size D
            Each entry represents a standard dev. vector for random walk proposal
            Can provide just one entry, or multiple entries.
            Each sample will generate via a sequence of
            Metropolis propose->decide transitions, one for each entry in list
        random_state : int or numpy.random.RandomState
            Initial state of this sampler's random number generator.
            Set deterministically for reproducability and debugging.
            If integer, will create a numpy PRNG with that as seed.
            If numpy PRNG object, will call its randn() and rand() methods.

        Returns
        -------
        New RandomWalkSampler object
        '''
        self.calc_tilde_log_pdf = calc_tilde_log_pdf
        assert isinstance(list_of_rw_stddev_D, list)
        self.D = list_of_rw_stddev_D[0].size
        for rw_s_D in list_of_rw_stddev_D:
            assert isinstance(rw_s_D, np.ndarray)
            assert rw_s_D.size == self.D
            assert rw_s_D.shape == (self.D,)
            assert np.all(rw_s_D >= 0.0)
        self.list_of_rw_stddev_D = list_of_rw_stddev_D

        if hasattr(random_state, 'rand'):
            self.random_state = random_state
        else:
            # Will raise error if not cast-able to int
            self.random_state = np.random.RandomState(int(random_state))


    def sample_standard_normal_vector(self):
        ''' Draw values from "standard" Normal with mean 0.0 and variance 1.0

        Uses internal pseudo-random number generator.

        Args
        ----
        None.

        Returns
        -------
        eps_D : 1D array, size (D,)
        '''
        return self.random_state.randn(self.D)


    def sample_uniform_0_to_1(self):
        ''' Draw scalar float from Uniform between 0.0 and 1.0

        Uses internal pseudo-random number generator to safely draw repeatable values.

        Args
        ----
        None.

        Returns
        -------
        rand_val : scalar float
        '''
        return self.random_state.rand(1)


    def draw_samples(self, zinit_D=None, n_keep_samples=1, n_burnin_samples=0):
        ''' Draw samples from target distribution via MCMC

        Args
        ----
        zinit_D : 1D array, size (D,)
            Initial state of the target random variable to sample.
        n_keep_samples : int
            Number of samples to generate that we intended to keep
        n_burnin_samples : int
            Number of samples to generate that wwe intend to discard

        Returns
        -------
        z_list : list of numpy arrays, each of size (D,)
            Each entry is a sample from the MCMC procedure.
            Will contain n_keep_samples entries.
        sample_info : dict
            Contains information about this MCMC chain's progress, including
            'accept_rate' : Number of accepted proposals across all iterations
            'accept_rate_keep' : Number of accepted proposals in kept phase
            'did_accept_SP' : binary array, size (S, P)
                Accept/reject status of every proposal
        '''
        K = int(n_keep_samples)
        B = int(n_burnin_samples)
        S = B + K

        z_D = 1.0 * zinit_D
        z_list = list()
        z_list.append(z_D)

        P = len(self.list_of_rw_stddev_D)
        n_accept = 0
        did_accept_SP = np.zeros((S, P), dtype=np.float64)
        for s in range(K +B):
            for p, rw_stddev_D in enumerate(self.list_of_rw_stddev_D):

                u_accept = self.sample_uniform_0_to_1()

                eps_D = self.sample_standard_normal_vector()
                
                # TODO construct proposed vector from z_D
                # Hint: Recall eps_D ~ [Norm(0.0, 1.0), ... Norm(0.0, 1.0)]
                # Want to transform eps_D to get zprime_D ~ N(z_D, rw_stddev_D)
                zprime_D = np.zeros(self.D) # TODO FIXME

                # TODO compute accept ratio
                # for proposed move from z_D to zprime_D
                A = 1.0 # TODO FIXME
                did_accept = True # TODO FIXME 
                if did_accept:
                    # Accepted proposal.
                    # TODO update the sampler state to its new value
                    znew_D = 1.0 * z_D # TODO FIXME

                    n_accept += 1
                    did_accept_SP[s, p] = 1
                else:
                    # Rejected. Keep old value.
                    znew_D = 1.0 * z_D

            z_D = znew_D
            z_list.append(z_D)

        z_keep_list = copy.deepcopy(z_list[(B+1):])
        info = dict(
            z_list=z_list,
            n_burnin_samples=n_burnin_samples,
            n_keep_samples=n_keep_samples,
            n_accept=n_accept,
            accept_rate=n_accept/float(S),
            accept_rate_keep=np.mean(did_accept_SP[B:,:]),
            did_accept_SP=did_accept_SP)
        return z_keep_list, info