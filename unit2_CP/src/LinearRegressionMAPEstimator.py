'''
Summary
-------
Defines a maximum a-posterior estimator for linear regression

MAPEstimator supports these API functions common to any sklearn-like regression model:
* fit
* predict
* score

Resources
---------
See COMP 136 course website for the complete problem description and all math details
'''

import numpy as np
import scipy.stats

class LinearRegressionMAPEstimator():
    """
    Maximum A-Posteriori Estimator for linear regression

    Attributes
    ----------
    feature_transformer : feature transformer
        Any transformer that implements "transform" and "get_feature_size"
        See provided FeatureTransformPolynomial.py in starter code
    alpha : float, must be positive
        Defines precision for the prior over the weights
        p(w) = Normal(0, alpha^{-1} I)
    beta : float, must be positive
        Defines precision for the likelihood of t_n given features x_n
        p(t_n | x_n, w) = Normal( w^T x_n, beta^{-1})

    Examples
    --------
    >>> N, D = 100, 1
    >>> prng = np.random.RandomState(0)
    >>> x_ND = prng.randn(N, D)
    >>> t_N = 5 * x_ND[:,0] + 1
    >>> t_N.shape == (N,)
    True

    >>> from FeatureTransformPolynomial import PolynomialFeatureTransform
    >>> txfm = PolynomialFeatureTransform(order=1, input_dim=D)

    >>> alpha = 1.0
    >>> beta = 20.0
    >>> map = LinearRegressionMAPEstimator(txfm, alpha, beta)
    >>> map = map.fit(x_ND, t_N)
    >>> map.w_map_M.shape
    (2,)
    >>> map.w_map_M
    array([0.99964554, 4.99756957])
    >>> txfm.get_feature_names()
    ['bias', 'x 0^1']
    """

    def __init__(self, feature_transformer=None, alpha=1.0, beta=1.0):
        self.feature_transformer = feature_transformer
        self.alpha = float(alpha)
        self.beta = float(beta)


    def get_params(self, deep=False):
        ''' Obtain key attributes for this object as a dictionary

        Needed for use with sklearn CV functionality

        Returns
        -------
        param_dict : dict
        '''
        return {'alpha': self.alpha, 'beta': self.beta, 'feature_transformer': self.feature_transformer}

    def set_params(self, **params_dict):
        ''' Set key attributes of this object from provided dictionary

        Needed for use with sklearn CV functionality

        Returns
        -------
        self. Internal attributes updated.
        '''
        for param_name, value in params_dict.items():
            setattr(self, param_name, value)
        return self

    def fit(self, x_ND, t_N):
        ''' Fit this estimator to provided training data

        Args
        ----
        x_ND : 2D array, shpae (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim
        t_N : 1D array, shape (N,)
            Outputs for regression

        Returns
        -------
        self. Internal attributes updated.
        '''
        M = self.feature_transformer.get_feature_size() # num features
        Phi_NM = self.feature_transformer.transform(x_ND)
        ## TODO update w_map_M attribute via formulas from Bishop
        self.w_map_M = np.zeros(M)
        return self


    def predict(self, x_ND):
        ''' Make predictions of output value for each provided input feature vectors

        Args
        ----
        x_ND : 2D array, shape (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim

        Returns
        -------
        t_est_N : 1D array, size (N,)
            Each entry at index n is prediction given features in row n
        '''
        phi_NM = self.feature_transformer.transform(x_ND)
        N, M = phi_NM.shape
        ## TODO compute mean prediction
        return np.zeros(N)

    def predict_variance(self, x_ND):
        ''' Produce predictive variance at each input feature

        Args
        ----
        x_ND : 2D array, shape (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim

        Returns
        -------
        t_var_N : 1D array, size (N,)
            Each entry at index n is variance given features in row n
        '''
        phi_NM = self.feature_transformer.transform(x_ND)
        N, M = phi_NM.shape
        ## TODO compute variance
        return 0.05 * np.ones(N)


    def score(self, x_ND, t_N):
        ''' Compute the average log probability of provided dataset
    
        Assumes w is set to MAP value (internal attribute).
        Assumes Normal iid likelihood with precision \beta.

        Args
        ----
        x_ND : 2D array, shape (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim
        t_N : 1D array, shape (N,)
            Outputs for regression

        Returns
        -------
        avg_log_proba : float
        '''
        N = x_ND.shape[0]
        mean_N = self.predict(x_ND)
        total_log_proba = scipy.stats.norm.logpdf(
            t_N, mean_N, 1.0/np.sqrt(self.beta))
        return np.sum(total_log_proba) / N










