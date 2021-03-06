"""
Provides basic quant functions to be reused.

"""

import numpy as np
import sklearn


def series_interp(series, method):
    """
    Returns an interpolated series for an input.

    Args:
        series (ndarray): Series input.
        method (str): Fill missing data method.

    Returns:
        (ndarray): Interpolate series output.
    """
    series = series.values
    if method == 'flat':
        for i in range(1, len(series)):
            if series[i] is None or series[i] == 0:
                series[i] = series[i - 1]
    elif method[:6] == 'interp':
        for i in range(0, len(series)):
            if np.isnan(series[i]) or series[i] == 0.0:
                pos_f = i
                pos_b = i
                while pos_f < len(series) and (np.isnan(series[pos_f]) or series[pos_f] == 0.0):
                    pos_f += 1
                while pos_b >= 0 and (np.isnan(series[pos_b]) or series[pos_b] == 0.0):
                    pos_b -= 1
                if pos_f == len(series) and (np.isnan(series[pos_f - 1]) or series[pos_f - 1] == 0.0):
                    series[i] = series[pos_b]
                elif pos_b == 0 and (np.isnan(series[pos_b]) or series[pos_b] == 0.0):
                    series[i] = series[pos_f + 1]
                else:
                    d = (i - pos_b) / (pos_f - pos_b)
                    if method[-3:] == 'lin' and pos_f < len(series):
                        series[i] = series[pos_f] * d + series[pos_b] * (1 - d)
                    elif method[-6:] == 'loglin' and pos_f < len(series):
                        series[i] = (series[pos_f] ** d) * (series[pos_b] ** (1 - d))
    return series


def levels_to_returns(hmd, method, horizon, overlapping, data_fill):
    """
    Converts a series of levels into returns.

    Args:
        hmd (pandas.core.series.Series): Input hmd data.
        method (str): 'absolute', 'relative', 'log'
        horizon (int): Day lag for returns.
        overlapping (bool): Use overlapping returns.
        data_fill (str): Fill missing data method.

    Returns:
        (pandas.core.frame.DataFrame): Dataframe containing returns for dates.
    """
    if overlapping:
        n = len(hmd)
        all_dates = []
        pos = horizon - n % horizon - 1
        for d in range(int(n / horizon)):
            all_dates.append(hmd.index[pos])
            pos += horizon
        hmd = hmd[hmd.index.isin(all_dates)]
        horizon = 1
    n = len(hmd)
    hmd = series_interp(hmd, data_fill)
    returns = []
    for i in range(n - horizon):
        if method == 'absolute':
            returns.append(hmd[i+horizon] - hmd[i])
        elif method == 'relative':
            returns.append(hmd[i+horizon] / hmd[i] - 1)
        elif method == 'log':
            returns.append(np.log(hmd[i+horizon] / hmd[i]))
    return returns


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


def stepwise_linear_regression(x, y, intercept, stepwise_steps=10):
    """
    Performs a forward stepwise regression.

    Args:
        x (ndarray): Features (inputs)
        y (ndarray): Labels (targets)
        intercept (bool): Include an intercept in the regression.
        stepwise_steps (int): Number of steps to use.
    """
    x = np.transpose(x)
    i = 0
    all_regressors = []
    while i < stepwise_steps:
        print('Step ' + str(i))
        all_scores = []
        for j in range(len(x)):
            regressors = all_regressors.copy()
            regressors.append(j)
            test_series = x[regressors]
            test_series = test_series.reshape(-1,  i + 1)
            all_scores.append(sklearn.linear_model.LinearRegression(fit_intercept=intercept).fit(test_series, y).
                              score(test_series, y))
        all_regressors.append([i for i, j in enumerate(all_scores) if j == max(all_scores)][0])
        i += 1
    f = sklearn.linear_model.LinearRegression(fit_intercept=intercept).fit(test_series, y)
    return f, f.score(test_series, y), f.coef_, all_regressors


def linear_regression(regression_type, x, y, intercept, alpha=0):
    """
    Returns sklearn regression results.

    Args:
        regression_type (str): 'ols', 'ridge', 'lasso'
        x (ndarray): Features (inputs)
        y (ndarray): Labels (targets)
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


def returns_to_bool(returns):
    """
    Convert a series of returns into a bool vector.

    Args:
        returns: Input vector containing float or int.

    Returns:

    """
    bools = []
    for i in returns:
        if i >= 0:
            bools.append(1)
        elif i < 0:
            bools.append(0)
    return bools


def string_to_none(string):
    """
    Convets a string input to none if that is the value

    Args:
        string: Value to convert

    Returns:

    """
    if string == 'None':
        string = None
    return string
