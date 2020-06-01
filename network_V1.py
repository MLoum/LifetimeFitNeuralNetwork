"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# Libraries
from scipy.special import expit
# Standard library
import random


# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import math


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.results = []

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a) + b)
        b, w = self.biases[-1], self.weights[-1]
        a = relu(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # print(np.shape(mini_batches))
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            #activation = relu(z)
            activations.append(activation)

        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = relu(z)
        activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
             relu_prime(zs[-1])
                #prime(zs[-1])
                #sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            #sp = relu_prime(z)
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
        # print(self.biases)
        # print(test_results)
        mean_error_square = 0
        for result in test_results:
            mean_error_square += (result[0] - result[1])**2
        mean_error_square = mean_error_square[0]
        mean_error_square /=  len(test_data)
        self.results.append(mean_error_square)
        return mean_error_square
        # return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        # print("coût : ",1/2*(output_activations - y)**2)
        return (output_activations - y)

    def last_epoch_assess(self, test_data, epochs, eta):
        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
        print("nb d'epochs :", epochs, "learning rate eta :", eta, "dimension du reseaux :", self.sizes)
        erreurS_relative = []
        tps_vie = []
        for result in test_results:
            erreur_relative = abs(result[0] - result[1])/result[1] * 100
            erreur_relative = erreur_relative[0]
            print("tps de vie vrai :", result[1], "tps de vie du réseau", result[0], "Erreur relative en % :", erreur_relative)
            erreurS_relative.append(erreur_relative)
            tps_vie.append(result[1][0])

        self.mean_erreur_relatives = np.mean(erreurS_relative)
        self.std_erreur_relatives = np.std(erreurS_relative)
        print("moyenne :" + str(self.mean_erreur_relatives) + "   std : " + str(self.std_erreur_relatives))
        plt.plot(np.arange(epochs), self.results)
        plt.xlabel("Numero epoch")
        plt.ylabel("Mean square error")
        plt.title("learning rate :" + str(eta) + "sizes :" + str(self.sizes))
        plt.savefig("apprentissage - epochs =" + str(epochs) + " - eta = " + str(eta) + " - sizes = " + str(self.sizes) + ".png")
        plt.show()
        plt.plot(tps_vie, erreurS_relative, "ro")
        plt.title("Correlation ?")
        plt.xlabel("Tps de vie")
        plt.ylabel("Erreur relative")
        plt.savefig("Correlation  - epochs =" + str(epochs) + " - eta = " + str(eta) + " - sizes = " + str(self.sizes) + ".png")
        plt.show()





#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return expit(z)


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    """fonction ReLU"""
    # return sigmoid(z)
    return z
    # return np.maximum(0, z)


def relu_prime(z):
    """dérivée ReLU"""
    # return sigmoid(z) * (1 - sigmoid(z))
    return 1
    # return np.heaviside(z, 0)


def prime(z):
    return np.ones(z.shape)
