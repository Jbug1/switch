import numpy as np


def minexp(z):
    """
    subtract all values from 1, then exponentiate

    makes most sense with gini? u tell me
    """

    return np.exp(1 - z)


def divexp(z):
    """
    divide 1 by all values in imps and exponentiate
    """
    # add jitter to avoid div 0
    return np.exp(1 / (z + 0.00001))


def sigmoid(z):

    return 1 / (1 + np.exp(-z))


def frac(z):

    return 1 / (-z + 0.001)


def thresh(z, thresh):

    if z > thresh:
        return z

    else:
        return 0


def poslog(z):

    return 1 / np.log(-z + 1.0001)


def negexp(z):

    return np.exp(z)


def lin(z):

    return 1.001 + z


def flat(z):

    return 1


#########################best n scale functions#############################
class best_n_scales:
    def linear(depth, best_n):

        return best_n - depth

    def log(depth, best_n):

        return best_n - np.log(depth)

    def square(depth, best_n):

        return best_n - depth**2

    def sqrt(depth, best_n):

        return best_n - depth**0.5

    def inv_linear(depth, best_n):

        return depth

    def inv_log(depth, best_n):

        return np.log(depth)

    def inv_square(depth, best_n):

        return depth**2

    def inv_sqrt(depth, best_n):

        return depth**0.5
