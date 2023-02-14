'''
Summary
-------
Defines a maximum likelihood estimator for unigrams

Supports a common API for unigram probability estimators:
* fit
* predict_proba
* score

Resources
---------
See CS 136 course website for complete problem description and all math details
'''

import numpy as np
from Vocabulary import Vocabulary


class MLEstimator():
    """
    Maximum Likelihood Estimator for unigram probabilities

    To avoid pathologies with unseen words,
    we perform some ad-hoc smoothing so that this estimator:
    * 1) is a valid PMF over the vocabulary
    * 2) does not give zero probability to any word

    Attributes
    ----------
    vocab : Vocabulary object
    epsilon_unseen_proba : float between 0.0 and 1.0
        Probability mass allowed for all unseen words

    Internal State Attributes (filled after fit)
    -------------------------
    total_count : int
        Total number of words observed in train word list
    count_V : 1D array of size V and type int
        Each entry gives count of corresponding vocab word in train word list
    unseen_count : int
        Total number of vocab terms that never appeared in train word list

    Examples
    --------
    ## Note: Will NOT pass with starter code
    >>> word_list = ['dinosaur', 'trex', 'dinosaur', 'stegosaurus', 'known_unseen']
    >>> mle = MLEstimator(Vocabulary(word_list), epsilon_unseen_proba=0.1)
    >>> mle.fit(word_list[:-1]) # all but last word
    >>> mle.predict_proba('dinosaur')
    0.45
    >>> mle.predict_proba('known_unseen')
    0.1
    >>> mle.predict_proba('UNKNOWN_unseen')
    Traceback (most recent call last):
    ...
    KeyError: 'Word UNKNOWN_unseen not in the vocabulary'
    """

    def __init__(self, vocab, epsilon_unseen_proba=0.000001):
        self.vocab = vocab
        self.epsilon_unseen_proba = epsilon_unseen_proba

        # State that is set after call to 'fit'
        self.total_count = 0
        self.unseen_count = 0
        self.count_V = None

    def fit(self, word_list):
        ''' Fit this estimator to provided training data

        Args
        ----
        word_list : list of str
            Each entry is a word that can be looked up in the vocabulary

        Returns
        -------
        None. Internal attributes updated.

        Post Condition
        --------------
        Attributes will updated based on provided word list
        * The 1D array count_V is set to the count of each vocabulary word
        * The integer total_count is set to the total length of the word list
        '''
        self.count_V = np.zeros(self.vocab.size)
        self.total_count = 0
        self.unseen_count = 0
        # TODO update total_count and unseen_count
        # TODO update the count_V array

    def predict_proba(self, word):
        ''' Predict probability of a given unigram under this model

        Assumes this word is in the vocabulary

        Args
        ----
        word : string
            Known word that can be looked up in the vocabulary

        Returns
        -------
        proba : float between 0 and 1

        Raises
        ------
        KeyError if the provided word is not in the vocabulary
        '''
        # TODO calculate estimated proba of the provided word
        # TODO use count_V, total_count, unseen_count (set after calling fit)
        return 1.0 / self.vocab.size  # TODO change this placeholder!

    def score(self, word_list):
        ''' Compute the average log probability of words in provided list

        Args
        ----
        word_list : list of str
            Each entry is a word that can be looked up in the vocabulary

        Returns
        -------
        avg_log_proba : float between (-np.inf, 0.0)
        '''
        total_log_proba = 0.0
        for word in word_list:
            total_log_proba += np.log(self.predict_proba(word))
        return total_log_proba / len(word_list)
