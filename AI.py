import numpy as np
import glob
from PIL import Image
from random import random
from random import seed
from math import *

one_hot_encoding = {
    'A': 0,
    'B': 5,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'M': 1
}


def create_data():
    data = []
    for filename in glob.glob('images/*.jpg'):
        img = Image.open(filename)
        arr = np.array(img)
        img_arr = []
        for i in range(0, 20):
            for j in range(0, 20):
                #Because the image is grayscale,
                # we can round the first pixel
                img_arr.append(round(arr[i][j][0] / 255, 2))
        img_arr.append(one_hot_encoding.get(filename[7]))
        data.append(img_arr)
    return data


def initialize(n_input, n_hidden, n_output):
    network = list()
    hidden_layer = [{'weights': [random() for _ in range(n_input + 1)]} for _ in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in range(n_output)]
    network.append(output_layer)
    return network


def neuron_sum(weights, inputs):
    act = weights[-1]
    for i in range(len(weights) - 1):
        act += weights[i] * inputs[i]
    return act


def sigmoid_activation(activation):
    return 1.0 / (1.0 + exp(-activation))


def feed_forward(network, row):
    inputs = row
    for l in network:
        new_inputs = []
        for neuron in l:
            activation = neuron_sum(neuron['weights'], inputs)
            neuron['output'] = sigmoid_activation(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def sigmoid_derivative(output):
    return output * (1 - output)


def backpropagation(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] + neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, data, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in data:
            outputs = feed_forward(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backpropagation(network, expected)
            update_weights(network, row, l_rate)
    print(' error=%.3f' % (sum_error))

network = list()

def train(l_rate, neurons):
    seed(1)
    dataset = create_data()
    n_inputs = 400
    n_outputs = 6
    network = initialize(n_inputs, neurons, n_outputs)
    train_network(network, dataset, l_rate, 200, n_outputs)


def predict():
    img = Image.open('test.jpg')
    arr = np.array(img)
    img_arr = []
    for i in range(0, 20):
        for j in range(0, 20):
            img_arr.append(round(arr[i][j][0] / 255, 2))
    outputs = feed_forward(network, img_arr)
    index = outputs.index(max(outputs))
    return list(one_hot_encoding.keys())[list(one_hot_encoding.values()).index(index)]
