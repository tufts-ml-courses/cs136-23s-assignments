'''
Identify Feature Transform

Summary
-------
Feature transform that just adds a constant "1" bias to original features

API
---
Follows an sklearn-like "transform" interface
* __init__()
* transform

'''

import numpy as np


class IdentityFeatureTransform:

    def __init__(self, input_dim=1, input_feature_names=None):
        ''' Create FeatureTransform that just does "identity' transform plus constant bias
        '''
        self.input_dim = int(input_dim)
        if input_feature_names is None:
            self.input_feature_names = ['x%2d' %
                                        a for a in range(self.input_dim)]
        else:
            self.input_feature_names = [str(a) for a in input_feature_names]

    def get_feature_size(self):
        ''' Get integer size of the transformed features

        Returns
        -------
        output_dim : int
        '''
        return self.input_dim + 1

    def get_feature_names(self):
        ''' Get list of string names, one for each transformed feature

        Returns
        -------
        f_names : list of strings

        Examples
        --------
        >>> tfm2 = IdentityFeatureTransform(input_dim=2, input_feature_names=['a', 'b'])
        >>> tfm2.get_feature_names()
        ['a', 'b', 'constant_bias_feature']
        '''
        return self.input_feature_names + ['constant_bias_feature']

    def transform(self, x_ND):
        ''' Perform feature transformation on raw input measurements

        Args
        ----
        x_ND : 2D array, shape (N, D)
                Each row is a D-dimension 'raw' feature vector

        Returns
        -------
        phi_NM : 2D array, shape (N, M)
                Each row is a M-dimension 'transformed' feature vector

        Examples
        --------
        >>> x_N2 = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        >>> tfm2 = IdentityFeatureTransform(input_dim=2, input_feature_names=['a', 'b'])
        >>> tfm2.transform(x_N2)
        array([[0., 0., 1.],
               [1., 1., 1.]])
        '''
        N, D = x_ND.shape
        assert self.input_dim == D
        phi_NM = np.zeros((N, D + 1), dtype=x_ND.dtype)
        phi_NM[:, :self.input_dim] = x_ND
        phi_NM[:, -1] = 1
        return phi_NM
