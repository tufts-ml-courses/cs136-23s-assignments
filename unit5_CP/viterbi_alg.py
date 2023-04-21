'''
Viterbi algorithm for Hidden Markov Models.

Includes:
* skeleton of run_viterbi_algorithm requiring edits
* a __main__ method with example input
* doctests showing expected behavior

Usage
-----
To see estimated state sequence on simple example printed to stdout, do
$ python viterbi_alg.py

To verify if the script passes the doctest tests, do
$ python -m doctest viterbi_alg.py

Examples
--------
>>> np.set_printoptions(precision=5, suppress=1)
>>> T = 10
>>> K = 2
>>> D = 4
>>> pi_K = np.ones(K) / K
>>> A_KK = (np.ones((K,K)) + 49.0 * np.eye(K)) / (49 + K)
>>> A_KK
array([[0.98039, 0.01961],
       [0.01961, 0.98039]])

## Create mean and stddev for each state and dim
# mean will be -1 in first state, and +1 in second state
# stddev always 1
>>> mu_KD = np.ones((K, D))
>>> mu_KD[0] *= -1
>>> stddev_KD = np.ones((K, D))

## Sample 'simple' dataset with T examples from state 0, then T more from state 1
>>> import scipy.stats
>>> prng = np.random.RandomState(0)
>>> x_state0_TD = prng.randn(T, D) * stddev_KD[0] + mu_KD[0]
>>> x_state1_TD = prng.randn(T, D) * stddev_KD[1] + mu_KD[1]
>>> x_TD = np.vstack([x_state0_TD, x_state1_TD])
>>> log_lik_TK = np.vstack([
... 	np.sum(scipy.stats.norm.logpdf(x_TD, mu_KD[k], stddev_KD[k]), axis=1)
... 	for k in range(K)]).T

## Run Viterbi algorithm
>>> zhat_T, log_pdf_x_and_z = run_viterbi_algorithm(np.log(pi_K), np.log(A_KK), log_lik_TK)

## Verify estimated hidden state sequence is correct
>>> zhat_T[:T]
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)

>>> zhat_T[T:]
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

## Verify computed log joint proba using HMM is better than a plain mixture model
>>> print("% .5f" % log_pdf_x_and_z)
-119.02517
>>> gmm_log_pdf_x_and_z = np.sum(np.log(pi_K[zhat_T]) + log_lik_TK[(np.arange(T+T), zhat_T)])
>>> print("% .5f" % gmm_log_pdf_x_and_z)
-127.90669
'''

import numpy as np


def run_viterbi_algorithm(log_pi_K, log_A_KK, log_lik_TK):
	''' Run Viterbi algorithm for a sequence given HMM params and precomputed likelihoods

	Uses dynamic programming to compute everything.
	Runtime is O(TK^2), which means the cost is:
	* linear in number of timesteps T
	* quadratic in number of discrete hidden states K

	Args
	----
	log_pi_K
		Entry k defined in math as:
		$$
			\log \pi_{k}
		$$
	log_A_KK
		Entry k,k defined in math as:
		$$
			\log A_{kk}
		$$
	log_lik_TK : 2D array, shape (T, K) = n_timesteps x n_states
		Array containing the data-given-state log likelihood at each timestep
		Entry t,k defined in math as:
		$$
			\log p(x_t | z_t = k, \phi)
		$$
		where $\phi$ represents the "emission" distribution parameters,
		which control generating data x_t given state z_t=k


	Returns
	-------
	zhat_T : 1D array, shape (T,)
		Array containing MAP estimate of hidden state sequence z_{1:T}
		Each entry must be an integer in {0, 1, 2, ... K-1}
		Defined in math:
		$$
			\hat{z}_{1:T} = \argmax_{z_1, \ldots z_T} p(z_{1:T} | x_{1:T}, \theta)
						  = \argmax_{z_1, \ldots z_T} p(z_{1:T}, x_{1:T} | \theta)
		$$
		where \theta = \pi, A, \phi

	log_pdf_x_and_zhat : float
		Scalar log probability density of joint config of zhat_{1:T} and x_{1:T}
		Defined as:
		$$
			\log p(x_{1:T}, z_{1:T} | \pi, A, \phi)
		$$
	'''
	T, K = log_lik_TK.shape

	# Allocate array for storing log joint probabilities
	log_w_TK = np.zeros((T,K))
	# Allocate array for storing "backpointers"
	b_TK = -1 * np.ones((T, K), dtype=np.int32)

	# TODO base case update of log_w_TK and b_TK at t=0
	for t in range(1, T):
		# TODO base case update of log_w_TK and b_TK at t=0
		log_w_TK[t, :] = 0.0
		b_TK[t, :] = 0

	# Allocate array for storing estimated z sequence
	zhat_T = np.zeros(T, dtype=np.int32)
	# TODO Update at final timestep
	zhat_T[-1] = 0

	for t in range(T-2, -1, -1): # count from T-2, .... 1, 0 inclusive
		# TODO Update at t given zhat at t+1
		zhat_T[t] = 0

	# TODO compute joint probability of entire sequence
	hmm_log_pdf_x_and_zhat = 0.0

	return zhat_T, hmm_log_pdf_x_and_zhat


if __name__ == '__main__':
	np.set_printoptions(precision=5, suppress=1)
	T = 10
	K = 2
	D = 4

	# Uniform initial state probability
	pi_K = np.ones(K) / K

	# Transition probabilities with strong self-transition bias
	A_KK = (np.ones((K,K)) + 49.0 * np.eye(K)) / (49 + K)
	print("--------------------------------")
	print("A_KK, with shape (K=%d,K=%d)" % (A_KK.shape[0], A_KK.shape[1]))
	print("--------------------------------")
	print(A_KK)

	# Create mean and stddev for each state and dim
	# mean will be -1 in first state, and +1 in second state
	# stddev always 1
	mu_KD = np.ones((K, D))
	mu_KD[0] *= -1
	stddev_KD = np.ones((K, D))

	# Sample 'simple' dataset with T examples from state 0, then T more from state 1
	import scipy.stats
	prng = np.random.RandomState(0)
	x_state0_TD = prng.randn(T, D) * stddev_KD[0] + mu_KD[0]
	x_state1_TD = prng.randn(T, D) * stddev_KD[1] + mu_KD[1]
	x_TD = np.vstack([x_state0_TD, x_state1_TD])

	# Compute likelihood of data-given-state at each timestep
	log_lik_TK = np.vstack([
		np.sum(scipy.stats.norm.logpdf(x_TD, mu_KD[k], stddev_KD[k]), axis=1)
		for k in range(K)]).T
	print("--------------------------------")
	print("log_lik_TK, with shape (T=%d,K=%d)" % (log_lik_TK.shape[0], log_lik_TK.shape[1]))
	print("--------------------------------")
	print(log_lik_TK)

	# Run the Viterbi algorithm
	zhat_T, log_pdf_x_and_zhat = run_viterbi_algorithm(np.log(pi_K), np.log(A_KK), log_lik_TK)

	print("--------------------------------")
	print("zhat_T, with shape (T=%d,)" % (zhat_T.shape[0]))
	print("--------------------------------")
	print(zhat_T)
