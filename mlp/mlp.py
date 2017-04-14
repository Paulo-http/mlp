import numpy as np

from helper import Helper

class MLP:

    def __init__(self, *args):
        self.helper = Helper()
        self.shape = args
        n = len(args)

        self.layers = []
        bias = 1
        self.layers.append(np.ones(self.shape[0] + bias))

        # Hidden layer(s) + output layer
        for i in range(1, n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n - 1):
            self.weights.append(np.zeros((self.layers[i].size, self.layers[i + 1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0, ] * len(self.weights)
        self.reset()

    #   Reset weights
    def reset(self):
        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size, self.layers[i + 1].size))
            self.weights[i][...] = (2 * Z - 1) * 0.25

    #   Propagate data from input layer to output layer.
    def propagate_forward(self, data):
        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1, len(self.shape)):
            # Propagate activity
            self.layers[i][...] = self.helper.sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]

    #   Back propagate error related to target using lrate.
    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error * self.helper.dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape) - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.helper.dsigmoid(self.layers[i])
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate * dw + momentum * self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error ** 2).sum()