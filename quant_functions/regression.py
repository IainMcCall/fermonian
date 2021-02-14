"""
Provides regression_fun() function to calculate a random forest.

"""
import logging

import numpy as np

from sklearn.model_selection import train_test_split
from quant_functions.basic import linear_regression, stepwise_linear_regression

logger = logging.getLogger('main')


def regression_fun(features, labels, regression_type, include_intercept, alpha, test_size=0.25, random_state=np.random,
                   stepwise_steps=10):
    """
    Creates a random forest from input features and labels. Outputs plots and updates estimates.

    Args:
        features (pandas.core.frame.DataFrame): Input data for the model (X).
        labels (pandas.core.series.Series): Target data for the model (Y).
        regression_type (str): 'ols', 'ridge', 'lasso', 'stepwise'
        include_intercept (bool): Include an intercept in the regression.
        alpha (float): Regularisation amount to apply in ridge or lasso.
        test_size (float) : % of out-of-sample data to use in test.
        random_state (int) : random_state for functions.
    Returns:
        (sklearn.ensemble.forest.RandomForestRegressor): Random forest output given the input.

    """
    logger.info('Starting to run regression')
    features = np.array(features)
    labels = np.array(labels)

    # Run test model
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size,
                                                                                random_state=random_state)
    if regression_type == 'stepwise':
        test_result, score, coef, regressors = stepwise_linear_regression(train_features, train_labels,
                                                                          include_intercept, stepwise_steps)
        test_features = np.transpose(test_features)
        test_features = test_features[regressors]
        test_features = test_features.reshape(-1, stepwise_steps)

        predictions = test_result.predict(test_features)
        sse = np.sum((predictions - test_labels) ** 2)
        mean_model_error = np.sqrt(np.mean((predictions - test_labels) ** 2))
        logger.info('Out-of-bag model error: ' + str(np.round(mean_model_error, 6) * 100) + '%')
        tss = np.sum((test_labels - np.mean(test_labels)) ** 2)
        result, score, coef, regressors = stepwise_linear_regression(features, labels, include_intercept,
                                                                     stepwise_steps)
    else:
        test_result, score, coef = linear_regression(regression_type, train_features, train_labels, include_intercept,
                                                     alpha)
        predictions = test_result.predict(test_features)
        sse = np.sum((predictions - test_labels) ** 2)
        mean_model_error = np.sqrt(np.mean((predictions - test_labels) ** 2))
        logger.info('Out-of-bag model error: ' + str(np.round(mean_model_error, 6) * 100) + '%')
        tss = np.sum((test_labels - np.mean(test_labels)) ** 2)
        result, score, coef = linear_regression(regression_type, features, labels, include_intercept, alpha)
        regressors = np.arange(0, features.shape[1], 1).tolist()
    return result, score, coef, mean_model_error, 1 - sse / tss, regressors
