import math
from weight import Weight

class Neuron:
    # Class Constructor
    # This method initializes all properties of this class.
    # @param layer: The Input layer that will be connected to the Hidden Neuron.
    # @param random: A random number.
    def __init__(self, *pargs):
        self.weights = None
        self.input = 0.0
        self.sigmoid = 1
        self.bias = 1.0
        self.error = 0.0
        self.learning_rate = 0.01
        self.output = None
        if pargs:
            inputs, rand = pargs
            self.weights = []
            for input in inputs.get_layer():
                w = Weight()
                w.input = input
                w.value = rand.uniform(-1, 1)
                self.weights.append(w)

    # Set the output of the neuron.
    def set_output_neuron(self, value):
        self.output = value

    # Linear combination implementation of the perceptron.
    def activate(self):
        self.input = 0.0
        for w in self.weights:
            self.input += w.value * w.input.output_neuron()

    # Activation function of the perceptron.
    def output_neuron(self):
        if self.output != None:
            return self.output
        return 1 / (1 + math.exp(-self.sigmoid * (self.input + self.bias)))

    # Calculates the error.
    def error_feedback(self, input):
        weight = None
        for w in self.weights:
            if w.input == input:
                weight = w
                break
        return self.error * self.derivative() * weight.value

    # The derivative of activation function.
    def derivative(self):
        return self.output_neuron() * (1 - self.output_neuron())

    # Adjust the weights connected to the neuron.
    def adjust_weights(self, value):
        self.error = value
        for w in self.weights:
            w.value += self.error * self.derivative() * self.learning_rate * w.input.output_neuron()
        self.bias += self.error * self.derivative() * self.learning_rate