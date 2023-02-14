'''
Summary
-------
This script produces a figure showing how several estimators perform at the task of
computing the log probability of heldout words (y-axis) as training set size increases (x-axis).

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.25)

from Vocabulary import Vocabulary, load_lowercase_word_list_from_file
from MLEstimator import MLEstimator
from MAPEstimator import MAPEstimator
from PosteriorPredictiveEstimator import PosteriorPredictiveEstimator

if __name__ == '__main__':
    alpha = 1.001
    epsilon_unseen_proba = 0.00001

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
    # We take fractions of a rounded number (400k) so sizes are easy to discuss
    Nbig = 400000
    frac_train_list = [1./128, 1./64, 1./32, 1./16, 1./8, 1./4, 1./2, 1.]
    n_train_list = [int(np.ceil(frac * Nbig)) for frac in frac_train_list]
    n_train_list[-1] = len(train_word_list) # largest size = all train data

    # Preallocate arrays to store the scores for each estimator
    mle_scores = -7.5 + np.zeros_like(frac_train_list)
    map_scores = -8.0 + np.zeros_like(frac_train_list)
    ppe_scores = -8.5 + np.zeros_like(frac_train_list)

    # TODO loop over all train set sizes in `n_train_list`
    # ---- fit ML Estimator on train, then score it on test set
    # ---- fit MAP Estimator on train, then score it on test set
    # ---- fit PosteriorPredictive Estimator on train, then score it on test set

    fig_h, ax_h = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(6, 4))
    arange_list = np.arange(len(n_train_list))
    ax_h.plot(arange_list, mle_scores, 'm.-', label='ML estimator')
    ax_h.plot(arange_list, map_scores, 'b.-', label='MAP estimator')
    ax_h.plot(arange_list, ppe_scores, 'g.-', label='PosteriorPred estimator')

    ax_h.set_xticks([a for a in arange_list[1::2]])
    ax_h.set_xticklabels(['%dk' % (a/1000) for a in n_train_list][1::2])
    ax_h.set_xlim([-0.05, 1.05*max(arange_list)])
    ax_h.set_ylim([-10.1, -6.6])

    plt.xlabel("TODO fill xlabel")
    plt.ylabel("TODO fill ylabel")
    plt.legend(loc='lower right')
    plt.tight_layout();
    plt.show()
