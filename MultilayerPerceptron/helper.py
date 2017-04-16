import csv
import random

class Helper:
    # Class Constructor
    # This method initializes all properties of this class.
    def __init__(self):
        self.config = [['iris.csv', 4, 3],
                       ['wine.csv', 13, 3],
                       ['cancer.csv', 9, 2],
                       ['quality.csv', 11, 11],
                       ['abalone.csv', 8, 29],
                       ['adult.csv', 14, 2]]
        self.filename = None
        self.attrs = None
        self.classes = None
        self.csv = None
        self.validations = None
        self.outputs = None

    #   load csv file and normalize the data
    def load_csv_file(self):
        array = []
        folder = "csv/"
        with open(folder + self.filename, 'rb') as csvfile:
            lines = csv.reader(csvfile, delimiter=';')
            data = list(lines)
            for x in range(len(data)):
                line = ()
                for y in range(self.attrs):
                    value = round(float(data[x][y]), 3)
                    line += (value,)
                array.append([line, int(data[x][-1])])
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

    #   add output neuron
    def add_output_neuron(self):
        for elements in self.csv:
            value = int(elements[-1])
            for x in range(self.classes):
                if value == x+1:
                    elements.append(1)
                else:
                    elements.append(0)
            idx = elements.index(value)
            del elements[idx]

    #   prepare list, make shuffle and 10-fold-cross validation
    def prepare_list(self, idx):
            self.filename = self.config[idx][0]
            self.attrs = self.config[idx][1]
            self.classes = self.config[idx][2]
            self.csv = self.load_csv_file()
            self.add_output_neuron()
            self.shuffle_csv_list()
            self.validations = self.cross_validation()

    def prepare_training(self, validation):
        validations = list(self.validations)
        validations.remove(validation)
        training = list()
        for part in validations:
            for value in part:
                training.append(value)
        return training

    def current_part(self, validation):
        return self.validations.index(validation) + 1

    def print_result(self, part, neurons, correctly_instances, absolute_error, squared_error, iterations):
        header = "\nPart, " \
                 "Number of neurons, " \
                 "Correctly classified instances (%), " \
                 "Mean absolute error, " \
                 "Root mean squared error, " \
                 "Number of interations"
        print header
        print part, neurons, correctly_instances, absolute_error, squared_error, iterations
