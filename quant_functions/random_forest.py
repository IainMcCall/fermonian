"""
Provides random_forest_fun() function to calculate a random forest.

"""
import logging

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger('main')


def random_forest_fun(features, labels, test_size=0.25, n_estimators=1000, criterion='mse', max_depth=None,
                      min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                      max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
                      oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0,
                      max_samples=None):
    """
    Creates a random forest from input features and labels. Outputs plots and updates estimates.

    Args:
        features (pandas.core.frame.DataFrame): Input data for the model (X).
        labels (pandas.core.series.Series): Target data for the model (Y).
        estimators (int) : n_estimators in RandomForestRegressor.
        test_size (float) : % of out-of-sample data to use in test.
        random_state (int) : random_state for functions.

    Returns:
        (sklearn.ensemble.forest.RandomForestRegressor): Random forest output given the input.

    """
    logger.info('Starting to create random forest')
    feature_list = list(features.columns)
    features = np.array(features)
    labels = np.array(labels)

    # Run test model
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size,
                                                                                random_state=random_state)
    rf_test = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                    min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                    max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                    min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score,
                                    n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                                    ccp_alpha=ccp_alpha, max_samples=max_samples)
    rf_test.fit(train_features, train_labels)
    predictions = rf_test.predict(test_features)
    sse = np.sum((predictions - test_labels) ** 2)
    mean_model_error = np.sqrt(np.mean((predictions - test_labels) ** 2))
    logger.info('Out-of-bag model error: ' + str(np.round(mean_model_error, 6) * 100) + '%')
    tss = np.sum((test_labels - np.mean(test_labels)) ** 2)

    # Run prediction model
    rf = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                               min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                               min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                               max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                               min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score,
                               n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                               ccp_alpha=ccp_alpha, max_samples=max_samples)
    rf.fit(features, labels)
    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 6)) for feature, importance in zip(feature_list, importances)]
    return rf, feature_importances, mean_model_error, 1 - sse / tss
