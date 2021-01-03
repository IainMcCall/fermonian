"""
Provides NeuralNetwork class which is used as a neural network calculator.

To be added:

1. Biases
2. Choice of activation function.
3. Allow up to n hidden layers.


"""

import numpy as np
import math


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