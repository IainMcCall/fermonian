"""
Provides random_forest_fun() function to calculate a random fores.

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pydot

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz


def random_forest_fun(features, labels, outdir, estimators=1000, test_size=0.25, random_state=42):
    """
    Creates a random forest from input features and labels. Outputs plots.

    Args:
        features (pandas.core.frame.DataFrame): Input data for the model (X).
        labels (pandas.core.series.Series): Target data for the model (Y).
        outdir (str): Output directory.
        estimators (int) : n_estimators in RandomForestRegressor.
        test_size (float) : % of out-of-sample data to use in test.
        random_state (int) : random_state for functions.

    Returns:
        (sklearn.ensemble.forest.RandomForestRegressor): Random forest output given the input.

    """
    print('Starting to create random forest')
    feature_list = list(features.columns)
    features = np.array(features)
    labels = np.array(labels)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size,
                                                                                random_state=random_state)
    rf = RandomForestRegressor(n_estimators=estimators, random_state=random_state)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2), '')
    mape = 100 * (errors / abs(test_labels))
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    tree = rf.estimators_[5]
    export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png(os.path.join(outdir, 'tree.png'))

    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # Plot importances
    plt.figure()
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    plt.style.use('fivethirtyeight')
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'importances.jpg'))

    return rf
