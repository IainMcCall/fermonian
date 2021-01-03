"""
    Provides basic quant functions to be reused.

"""

import numpy as np
import pandas as pd


def levels_to_returns(data, method, horizon):
    """
    Converts a dataframe of levels into returns.

    Args:
        data (pandas.core.frame.DataFrame): Input dataframe with dates in index and series in columns.
        method (str): 'absolute', 'relative', 'log'
        horizon (int): Day lag for returns.

    Returns:
        (pandas.core.frame.DataFrame): Dataframe containing returns for dates.
    """
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