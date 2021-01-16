"""
Provides basic quant functions to be reused.

"""

import numpy as np
import pandas as pd
import sklearn


def levels_to_returns(data, method, horizon, overlapping):
    """
    Converts a dataframe of levels into returns.

    Args:
        data (pandas.core.frame.DataFrame): Input dataframe with dates in index and series in columns.
        method (str): 'absolute', 'relative', 'log'
        horizon (int): Day lag for returns.
        overlapping (bool): Is the data overlapping?

    Returns:
        (pandas.core.frame.DataFrame): Dataframe containing returns for dates.
    """
    if overlapping:
        n = len(data)
        all_dates = []
        pos = horizon - n % horizon - 1
        for d in range(int(n / horizon)):
            all_dates.append(data.index[pos])
            pos += horizon
        data = data[data.index.isin(all_dates)]
        horizon = 1
    all_returns = pd.DataFrame(index=data.index[horizon:])
    n = len(data)
    for c in data:
        series = data[c]
        returns = []
        for i in range(n - horizon):
            if method == 'absolute':
                returns.append(series[i+horizon] - series[i])
            elif method == 'relative':
                returns.append(series[i+horizon] / series[i] - 1)
            elif method == 'log':
                returns.append(np.log(series[i+horizon] / series[i]))
        all_returns[c] = returns
    return all_returns


def linear_interpolate_1d(x, y, x_new):
    y_new = []
    for t in x_new:
        if t <= x[0]:
            y_new.append(y[0])
        elif t >= x[-1]:
            y_new.append(y[-1])
        else:
            p = 0
            while t > x[p]:
                p += 1
            y_new.append(y[p] * (t - x[p-1]) / (x[p] - x[p-1]) + y[p-1] * (x[p] - t) / (x[p] - x[p-1]))
    return y_new


def regression_fun(regression_type, x, y, intercept, alpha=0):
    """
    Returns sklearn regression results.

    Args:
        regression_type (str): 'ols', 'ridge', 'lasso'
        x (pandas.core.frame.DataFrame): Features (inputs)
        y (list): Labels (targets)
        intercept (bool): Include an intercept in the regression.
        alpha (float): Regularisation amount to apply in ridge or lasso.
    """
    if regression_type == 'ols':
        f = sklearn.linear_model.LinearRegression(fit_intercept=intercept).fit(x, y)
    elif regression_type == 'lasso':
        alpha = np.mean(np.std(x)) * alpha
        f = sklearn.linear_model.Lasso(fit_intercept=intercept, alpha=alpha).fit(x, y)
    elif regression_type == 'ridge':
        alpha = np.mean(np.std(x)) * np.sqrt(len(y) * alpha)
        f = sklearn.linear_model.Ridge(fit_intercept=intercept, alpha=alpha).fit(x, y)
    return f, f.score(x, y), f.coef_
