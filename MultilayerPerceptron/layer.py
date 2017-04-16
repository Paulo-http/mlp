from neuron import Neuron

class Layer:
    # Class Constructor
    # This method initializes all properties of this class.
    # @param size: The number of Neurons that composes the layer.
    # @param layer: The Input layer that will be connected to the Hidden Neuron.
    # @param random: A random number.
    def __init__(self, size, *pargs):
        self.base = []
        if pargs:
            layer, rand = pargs
            for i in range(size):
                self.base.append(Neuron(layer, rand))
        else:
            for i in range(size):
                self.base.append(Neuron())

    def get_layer(self):
        return self.base
