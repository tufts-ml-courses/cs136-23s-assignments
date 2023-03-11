'''
Polynomial Feature Transform

Summary
-------
Feature transform that takes a vector and returns an order-r polynomial transform. 

The user can specify the order (0, 1, 2, 3, ...)

Example input: [x, y, z]
Example output (order = 1): [1 x^1 y^1 z^1] = [1 x y z]
Example output (order = 2): [1 x^1 y^1 z^1 x^2 y^2 z^2]
Example output (order = 3): [1 x^1 y^1 z^1 x^2 y^2 z^2 x^3 y^3 z^3]

API
---
Follows an sklearn-like "transform" interface
* __init__(<<options>>) : constructor
* transform(x) : transform provided data in array x
'''

import numpy as np


class PolynomialFeatureTransform:

    def __init__(self, order=1, input_dim=1, input_feature_names=None):
        ''' Create FeatureTransform that does polynomial transform of each feature up to given order
        '''
        if order < 0:
            raise ValueError("Polynomial order cannot be negative")
        if input_dim < 1:
            raise ValueError(
                "Input dimension must be an integer greater than or equal to 1")
        self.input_dim = int(input_dim)
        self.order = int(order)
        self.output_dim = self.input_dim * order + 1
        if input_feature_names is None:
            self.input_feature_names = ['x%2d' %
                                        a for a in range(self.input_dim)]
        else:
            self.input_feature_names = [str(a) for a in input_feature_names]

    def get_feature_size(self):
        return self.output_dim
        
    def get_feature_names(self):
        ''' Get list of string names, one for each transformed feature

        Examples
        --------
        >>> tfm2 = PolynomialFeatureTransform(order=3, input_dim=2, input_feature_names=['a', 'b'])
        >>> tfm2.get_feature_names()
        ['bias', 'a^1', 'b^1', 'a^2', 'b^2', 'a^3', 'b^3']
        '''
        feat_names = ['bias']
        for P in range(1, self.order+1):
            feat_names += map(lambda s: '%s^%d' % (s,P),
                              self.input_feature_names)
        return feat_names

    def transform(self, x_ND):
        ''' Perform feature transformation on raw input measurements

        Examples
        --------
        >>> x_ND = np.arange(0, 5)[:, np.newaxis]
        >>> x_ND
        array([[0],
               [1],
               [2],
               [3],
               [4]])
        >>> tfm0 = PolynomialFeatureTransform(order=0)
        >>> tfm0.transform(x_ND)
        array([[1],
               [1],
               [1],
               [1],
               [1]])
        >>> tfm1 = PolynomialFeatureTransform(order=1)
        >>> tfm1.transform(x_ND)
        array([[1, 0],
               [1, 1],
               [1, 2],
               [1, 3],
               [1, 4]])
        >>> tfm2 = PolynomialFeatureTransform(order=2)
        >>> tfm2.transform(x_ND)
        array([[ 1,  0,  0],
               [ 1,  1,  1],
               [ 1,  2,  4],
               [ 1,  3,  9],
               [ 1,  4, 16]])
        '''
        N, D = x_ND.shape
        if not self.input_dim == D:
            raise ValueError(
                "Mismatched input dimension. Expected %d but received %d" % (self.input_dim, D))

        phi_NM = np.zeros((N, self.output_dim), dtype=x_ND.dtype)

        phi_NM[:, 0] = 1.0
        m_dim = 1
        for P in range(1, self.order+1):
            phi_NM[:, m_dim:(m_dim + D)] = np.power(x_ND, P)
            m_dim += D
        return phi_NM
