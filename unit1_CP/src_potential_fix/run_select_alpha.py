'''
Summary
-------
This script produces a figure showing how the training set evidence varies (y-axis) as we
consider different alpha values (x-axis) for the Dirichlet prior of our model.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.0)

from Vocabulary import Vocabulary, load_lowercase_word_list_from_file
from MAPEstimator import MAPEstimator
from PosteriorPredictiveEstimator import PosteriorPredictiveEstimator

from scipy.special import gammaln


def calc_per_word_log_evidence(estimator, word_list):
    ''' Evaluate the log of the evidence, averaged per word

    Assumes the Dirichlet-Categorical model, marginalizing out the parameter

    Args
    ----
    estimator : PosteriorPredictiveEstimator
            Defines a Dir-Mult model
    word_list : list of strings
            Assumed that each string is in the vocabulary of the estimator

    Returns
    -------
    log_proba : scalar float
            Represents value of log p(word_list | alpha) / N
            where N = len(word_list)
            This marginalizes out the probability parameters of the Dir-Cat

    Examples
    --------
    >>> vocab4 = Vocabulary(["a", "b", "c", "d"])
    >>> est = PosteriorPredictiveEstimator(vocab=vocab4, alpha=1.0)
    >>> np.round(np.exp(calc_per_word_log_evidence(est, ["a"])), 5)
    0.25

    >>> vocab3 = Vocabulary(["a", "b", "c"])
    >>> est = PosteriorPredictiveEstimator(vocab=vocab3, alpha=0.1)
    >>> log_ev = calc_per_word_log_evidence(est, ["a", "a", "b", "b", "c", "c"])
    >>> np.round(np.exp(log_ev), 5)
    0.16438
    >>> log_ev = calc_per_word_log_evidence(est, ["c", "c", "c", "c", "c", "c"])
    >>> np.round(np.exp(log_ev), 5)
    0.77812
    >>> log_ev = calc_per_word_log_evidence(est, ["a", "a", "a", "a", "a", "a"])
    >>> np.round(np.exp(log_ev), 5)
    0.77812
    '''
    assert isinstance(estimator, PosteriorPredictiveEstimator)

    # TODO Fit the estimator to the words
    # TODO Calculate the log evidence using provided formulas in CP1 spec
    log_evidence = 0.0

    # Return the per word log evidence
    return log_evidence / float(len(word_list))

if __name__ == '__main__':
    # Load list of all filenames from 1945-Truman.txt to 2016-Obama.txt
    data_fpaths = [fpath for fpath in sorted(
        glob.glob("../state_of_union_1945-2016/*.txt"))]

    train_fpaths = data_fpaths[:-1] # train on prev speeches
    test_fpaths = data_fpaths[-1:]  # test on last year of Obama's 2nd term
    print("Testing on:")
    for f in test_fpaths:
        print(f)
    train_word_list = load_lowercase_word_list_from_file(train_fpaths)
    test_word_list = load_lowercase_word_list_from_file(test_fpaths)

    # Create vocabulary from all words
    all_word_list = train_word_list + test_word_list
    vocab = Vocabulary(all_word_list)
    print("Using vocab of size %d" % vocab.size)

    # Shuffle words in train set
    # Goal: remove chance that words from same year are adjacent
    #       so when we use smaller train sets, we aren't using specific years
    prng = np.random.RandomState(2016)
    prng.shuffle(train_word_list)

    # Make list of train-set sizes we will assess
    frac_train_list = [1./128, 1./16, 1.]
    n_train_list = [int(np.ceil(frac * len(train_word_list)))
                    for frac in frac_train_list]

    # Make list of alpha values we will assess, from 10^-2 to 10^3
    alpha_list = np.logspace(-2, 3, 11)

    fig_handle, ax_grid = plt.subplots(
        nrows=1, ncols=len(n_train_list), figsize=(14, 3),
        squeeze=True, sharex=True, sharey=True)

    for nn, N in enumerate(n_train_list):
        print("Evaluating with N = %d ..." % (N))

        train_log_ev_list = -8.5 + np.zeros_like(alpha_list)
        test_log_lik_list = -8.0 + np.zeros_like(alpha_list)

        # TODO fit an estimator to each alpha value
        # TODO evaluate training set's log evidence at each alpha value
        # TODO evaluate test set's estimated probability via estimator's 'score'

        best_ii_test = np.argmax(test_log_lik_list)
        best_ii_train = np.argmax(train_log_ev_list)
        best_alpha_test = alpha_list[best_ii_test]
        best_alpha_ev = alpha_list[best_ii_train]

        arange_list = np.arange(len(alpha_list))
        ax_grid[nn].plot(arange_list, test_log_lik_list, 'b.-',
            label=r'test log lik.: best $\alpha$ = %.2f' % best_alpha_test)
        ax_grid[nn].plot(arange_list, train_log_ev_list, 'k.-',
            label=r'train log ev.: best $\alpha$ = %.2f' % best_alpha_ev)

        ax_grid[nn].legend(loc="lower left")

        ax_grid[nn].set_xticks(arange_list[::2])
        ax_grid[nn].set_xticklabels([
            ('% .2f' % a).replace('.00', '') for a in alpha_list[::2]])
        ax_grid[nn].set_title('train size N = %d' % N)
        ax_grid[nn].set_ylim([-10.1, -6.6])

        # TODO add appropriate labels
        ax_grid[nn].set_xlabel('TODO label x axis')
        if nn == 0:
            ax_grid[nn].set_ylabel('TODO label y axis')
    plt.tight_layout()
    plt.show()
