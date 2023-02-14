'''
Summary
-------
This script demos how to use the MLEstimator class
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
    # Load list of all filenames from 1945-Truman.txt to 2016-Obama.txt
    data_fpaths = [fpath for fpath in sorted(
        glob.glob("../state_of_union_1945-2016/*.txt"))]

    train_fpaths = data_fpaths[-8:-1] # train on Obama's first 7 speeches
    test_fpaths = data_fpaths[-1:]  # test on last year of Obama's 2nd term
    train_word_list = load_lowercase_word_list_from_file(train_fpaths)
    test_word_list = load_lowercase_word_list_from_file(test_fpaths)
    print("Using train word list of size %d" % len(train_word_list))
    print("Using test word list of size %d" % len(test_word_list))

    # Create vocabulary from all words
    all_word_list = train_word_list + test_word_list
    vocab = Vocabulary(all_word_list)
    print("Using vocab of size %d" % vocab.size)

    mleEst = MLEstimator(vocab, 0.0001)
    mleEst.fit(train_word_list)
    for test_word in test_word_list[:10]:
        proba = mleEst.predict_proba(test_word)
        print("%.5f = Pr(word = '%s' )" % (
            proba, test_word
            ))
    print("...")
    for test_word in test_word_list[-10:]:
        proba = mleEst.predict_proba(test_word)
        print("%.5f = Pr(word = '%s' )" % (
            proba, test_word
            ))
