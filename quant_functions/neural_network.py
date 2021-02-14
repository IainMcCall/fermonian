"""
Provides NeuralNetwork class which is used as a neural network calculator.

To be added:

1. Biases
2. Choice of activation function.
3. Allow up to n hidden layers.


"""
import logging
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from quant_functions.basic import returns_to_bool

logger = logging.getLogger('main')


def sklearn_ann(features, labels, test_size, hidden_layer_sizes=(100,), activation='logistic', solver='adam', alpha=0.0001,
                batch_size='auto', learning_rate='constant', learning_rate_init=0.0001,  power_t=0.5,
                max_iter=10000, shuffle=True, random_state=None, tol=1e-4, warm_start=False, momentum=0.9,
                nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
    """
    Given historical returns, a bool is output for the most likely direction of the return on the next day.

    Returns:

    """
    # Convert inputs into bools
    for x in features:
        features[x] = returns_to_bool(features[x])
    labels = returns_to_bool(labels)
    features = np.array(features).copy()
    labels = np.array(labels).copy()

    # Run test model
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size,
                                                                                random_state=random_state)
    ann_test = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                             batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                             power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol,
                             warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum,
                             early_stopping=early_stopping, validation_fraction=validation_fraction,
                             beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change,
                             max_fun=max_fun).fit(test_features, test_labels).fit(train_features, train_labels)
    predictions = ann_test.predict(test_features)
    success_rate = np.sum(abs(predictions - test_labels)) / len(test_labels)
    logger.info('Out-of-bag success rate: ' + str(np.round(success_rate, 6) * 100) + '%')

    # Run prediction model
    ann = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                        batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                        power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol,
                        warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum,
                        early_stopping=early_stopping, validation_fraction=validation_fraction,
                        beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change,
                        max_fun=max_fun).fit(test_features, test_labels).fit(features, labels)
    return ann, success_rate


def sigmoid(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = 1 / (1 + math.exp(-x[i, j]))
    return x


def sigmoid_derivative(y):
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i][j] = y[i][j] * (1 - y[i][j])
    return y


class NeuralNetwork:
    """
    Provides functions to perform feed-forward and back-propagation in a neural network.
    """
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], self.input.shape[0])
        self.weights2 = np.random.rand(self.input.shape[0], 1)
        self.y = y.reshape(-1, 1)
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        self.error = np.sum((self.y - self.output) ** 2)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2
        errors = self.y - self.output
        z2 = sigmoid(np.dot(self.layer1, self.weights2))
        d_weights2 = np.dot(self.layer1.T, (2 * errors * sigmoid_derivative(z2)))

        # application of the chain rule to find derivative of the loss function with respect to weights1
        z1 = sigmoid(np.dot(self.input, self.weights1))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * errors * sigmoid_derivative(z2), self.weights2.T)
                                           * sigmoid_derivative(z1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2