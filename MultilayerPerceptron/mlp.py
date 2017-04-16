import random
import math

from layer import Layer
from neuron import Neuron

class MLP:
    # Class Constructor
    # This method initializes all properties of this class.
    # @param iterations: The number of iterations.
    # @param architecture: (Number of hidden neurons, Number of input Neurons)
    # @param patterns: Collection of training patterns
    def __init__(self, patterns, iterations, architecture):
        self.hidden_dims, self.input_dims = architecture
        self.iteration = 0
        self.iterations = iterations
        self.patterns = list(patterns)
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.hidden_layer = None
        self.input_layer = None
        self.output = None
        self.theta = 0.01
        self.rand = random.Random()
        self.initialize()

    # Initialize the network based on its architecture.
    def initialize(self):
        self.input_layer = Layer(self.input_dims)
        self.hidden_layer = Layer(self.hidden_dims, self.input_layer, self.rand)
        self.output = Neuron(self.hidden_layer, self.rand)
        self.iteration = 0

    # Adjust the network weights.
    def adjust_weights(self, delta):
        self.output.adjust_weights(delta)
        for neuron in self.hidden_layer.get_layer():
            neuron.adjust_weights(self.output.error_feedback(neuron))

    # Propagates the perceptron and calculate the output.
    def activate(self, pattern):
        for i in range(len(pattern[0])):
            self.input_layer.get_layer()[i].set_output_neuron(pattern[0][i])
        for neuron in self.hidden_layer.get_layer():
            neuron.activate()
        self.output.activate()
        return self.output.output_neuron()

    # Do the training of the perceptron network.
    def train(self):
        error = 1.0
        while error > self.theta:
            error = 0.0
            for pattern in self.patterns:
                delta = pattern[1] - self.activate(pattern)
                self.absolute_error += delta
                self.adjust_weights(delta)
                error += math.pow(delta, 2)
                self.squared_error += error
            self.iteration += 1
            if self.iteration > self.iterations:
                self.initialize()
        self.absolute_error = self.absolute_error/self.iteration
        self.squared_error = self.squared_error/self.iteration

    # Test the network after trained.
    def execute(self, pattern):
        return self.activate(pattern)