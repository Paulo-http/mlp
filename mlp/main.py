import numpy as np

from helper import Helper
from mlp import MLP

if __name__ == '__main__':
    # Config
    helper = Helper()
    helper.prepare_list()

    network = MLP(2, 2, 1)
    samples = np.zeros(4, dtype=[('input', float, 2), ('output', float, 1)])

    # Example 1 : OR logical function
    print "Learning the OR logical function"
    network.reset()
    samples[0] = (0, 0), 0
    samples[1] = (1, 0), 1
    samples[2] = (0, 1), 1
    samples[3] = (1, 1), 1
    helper.learn(network, samples)

    # Example 2 : AND logical function
    print "Learning the AND logical function"
    network.reset()
    samples[0] = (0, 0), 0
    samples[1] = (1, 0), 0
    samples[2] = (0, 1), 0
    samples[3] = (1, 1), 1
    helper.learn(network, samples)

    # Example 3 : XOR logical function
    print "Learning the XOR logical function"
    network.reset()
    samples[0] = (0, 0), 0
    samples[1] = (1, 0), 1
    samples[2] = (0, 1), 1
    samples[3] = (1, 1), 0
    helper.learn(network, samples)