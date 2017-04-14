import numpy as np
import csv
import random
import math
import operator
import time

class Helper:

    def __init__(self):
        self.config = [['iris.csv', 4, 3]]

        self.filename = None
        self.attrs = None
        self.classes = None
        self.csv = None
        self.validation = None

        self.epochs = 2500
        self.lrate = .1
        self.momentum = 0.1

    def load_csv_file(self):
        array = []
        folder = "csv/"
        with open(folder + self.filename, 'rb') as csvfile:
            lines = csv.reader(csvfile, delimiter=';')
            data = list(lines)
            for x in range(len(data)):
                for y in range(self.attrs):
                    value = round(float(data[x][y]), 3)
                    data[x][y] = value
                array.append(data[x])
        return array

    #   prepare 10-fold-cross validation
    def cross_validation(self):
        k_fold = 10
        validation = []

        for div in xrange(0, k_fold):
            array = []
            lenght = range(len(self.csv) / 10)
            for x in lenght:
                array.append(self.csv[x + len(lenght) * div])
            validation.append(array)

        return validation

    def shuffle_csv_list(self):
        return random.shuffle(self.csv)

    def prepare_list(self):
        for idx in range(len(self.config)):
            start = time.time()

            self.filename = self.config[idx][0]
            self.attrs = self.config[idx][1]
            self.classes = self.config[idx][2]
            self.csv = self.load_csv_file()
            print self.csv
            self.shuffle_csv_list()
            self.validation = self.cross_validation()

            end = time.time()
            print("\ntime in %s: %s seconds\n" % (self.filename, (end - start)))

    #   Sigmoid like function using tanh
    def sigmoid(self, x):
        return np.tanh(x)

    #   Derivative of sigmoid above
    def dsigmoid(self, x):
        return 1.0 - x ** 2

    def learn(self, network, samples):
        # Train
        for i in range(self.epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward(samples['input'][n])
            network.propagate_backward(samples['output'][n], self.lrate, self.momentum)
        # Test
        for i in range(samples.size):
            o = network.propagate_forward(samples['input'][i])
            print i, samples['input'][i], '%.2f' % o[0],
            print '(expected %.2f)' % samples['output'][i]
        print