"""
Provides functions to calculate portfolio metrics.

"""
import numpy as np
from scipy.optimize import minimize


def portfolio_return(weights, returns):
    """

    Args:
        returns:
        weights:

    Returns:

    """
    return np.matmul(returns, weights)


def portfolio_stdev(weights, covar):
    """

    Args:
        weights:
        covar:

    Returns:

    """
    return np.dot(weights, np.dot(covar, weights))


def target_return(weights, returns, target):
    return portfolio_return(weights, returns) - target


def sum_weights(weights):
    return np.sum(weights) - 1


def markov_weights(returns, target):
    """
    Calculates the markov minimum variance weights for a given target return.

    Returns:

    """
    n = len(returns.columns)
    weight_0 = np.array([1 / n] * n)
    avg_returns = [np.average(returns[x]) for x in returns]
    covariance = returns.cov()
    cons = ({'type': 'eq', 'fun': target_return, 'args': (avg_returns, target)},
            {'type': 'eq', 'fun': sum_weights})
    bnds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    calibration = minimize(fun=portfolio_stdev, x0=weight_0, bounds=bnds, args=covariance, method='SLSQP', constraints=cons)
    return calibration
