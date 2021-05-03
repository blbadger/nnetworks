#! fcnetwork2.py
# A fully connected feed-forward neural network from scratch 
# (allowing for numpy & random).  Stochastic gradient descent 
# cost function minimization, the option of ReLu or sigmoid 
# activation functions, learning rate decay, and optional minibatch
# bootstrapping.

import random
import numpy as np 

class FullNetwork:

	def __init__(self, architecture, activation_function):

		# initialize biases
		self.biases = np.random.rand(100, 10)

		self.weights = []
		architecture = [784, 100, 10]
		for i in range(1, 3):
			j = architecture[i]
			k = architecture[i-1]
			self.weights.append(np.random.randn(k, j))

		self.weights = np.array(self.weights)
		self.activation = activation_function


	def activation_function(self, z):
		# Activation function, rectified linear unit
		z = np.array(z)
		if self.activation == 'relu':
			return max(0, z)

		if self.activation == 'sigmoid':
			return 1 / (1 + np.exp(-z))


	def activation_prime(self, z):
		# derivative of ReLu function
		if self.activation == 'relu':
			if z == 0:
				return 0
			else:
				return 1

		if self.activation == 'sigmoid':
			return activation_function(z)*(1 - activation_function(z))


	def network_output(self, ouput):
		# Feed-forward network output
		for index in range(item_length):
			weight, bias = self.weights[i], self.biases[i]
			output = self.activation_function(np.dot(weight, output) + bias)

		return output


	def gradient_descent(self, data, epochs, minibatch_size, learning_rate, bootstrap=False):
		# Trains neural network using stochastic gradient descent 
		# with optional bootstrapping.
		data = list(data)
		for epoch in range(epochs):
			if bootstrap:
				minibatch_number = len(data) // minibatch_size + 1
				minibatches = [random.sample(data, minibatch_size) for i in range(minibatch_number)]

			else:
				random.shuffle(data)
				minibatches = []
				i = 0
				while i < len(data):
					minibatches.append(np.array([j for j in data[i:i + minibatch_size]]))
					i += minibatch_size

			for minibatch in minibatches:
				self.update_minibatches(minibatch, learning_rate)

			print ('Epoch {0:01d} complete: {0:02d} /  \
				{0:03d} '.format(epoch, self.evaluate(data), len(data)))



	def update_minibatches(self, minibatch, learning_rate):
		# Apply gradient descent to update weights and biases during
		# minibatch training.
		partial_biases = [np.zeros(bias.shape) for bias in self.biases]
		partial_weights = [np.zeros(weight.shape) for weight in self.weights]

		delta_partial_biases, delta_partial_weights = self.backpropegation(minibatch)

		# udate weights and biases matrix-wise
		partial_biases += delta_partial_biases
		partial_weights += delta_partial_weights

		# update weights and biases
		self.biases = [bias - learning_rate * (partial_biases/len(minibatch))
						for bias, partial_bias in zip(self.biases, partial_biases)]

		self.weights = [weight - learning_rate * (partial_weights/len(minibatch))
						for weight, partial_weight in zip(self.weights, partial_weights)]


	def backpropegation(self, minibatch):
		# Compute the cost function gradient in terms
		# of weights and biases for each neuron

		d_partial_biases = np.array([[np.zeros(b.shape) for b in self.biases] for i in minibatch])
		d_partial_weights = np.array([[np.zeros(w.shape) for w in self.weights] for i in minibatch])

		# forward propegate to find neuron output vectors
		activation = np.array([i[0] for i in minibatch]) # first layer activation for all minibatch elements, 10x784 matrix
		
		activations = [activation] # array of subsequent activations for all minibatch elements
		z_vectors = [] # array of weighted activations + biases = z vectors for all mb elements

		minibatch_biases = np.array([self.biases for i in minibatch])
		minibatch_weights = np.array([self.weights for i in minibatch])

		for i in range(3):
			weight_matrix = np.array([weight for i in self.weights])
			bias_matrix = np.array([bias for i in self.biases])

			z = np.dot(weight_matrix.transpose(), activation) + bias_matrix[0][0]
			z_vectors.append(z)
			activation = self.activation_function(z_vectors)
			activations.append(activation)

		# ouput layer error calculation
		error = self.cost_function_derivative(activations[-1], activations[1]) \
		 * self.activation_prime(z_vectors[-1])

		d_partial_biases[-1] = error
		d_partial_weights = np.dot(error, activations[-2].transpose())

		# backpropegate through network
		for layer in range(2, 3):

			# increments from -2 to -len(network)
			z = z_vectors[-layer]
			cost_prime = self.activation_prime(z)
			error = np.dot(self.weights[-layer+1]/transpose(), error) * cost_prime

			d_partial_biases[-layer] = error
			d_partial_weights[-layer] = np.dot(error, activations[-layer-1].transpose())

	
		dp_biases = sum([i for i in d_partial_biases])
		dp_weights = sum([i for i in d_partial_weights])

		return d_bias, d_weight



	def evaluate(self, data):
		# Evaluate network performance on dataset

		results = [max(self.network_output(x), y) for x, y in data]
		return sum([i for i in results if int(x) == y])


	def cost_function_derivative(self, output_activations, y):
		return output_activations - y


import mnist_loader


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# network architecture: 784, 100, 10

net = FullNetwork([784, 100, 10], activation_function='sigmoid')

net.gradient_descent(training_data, 5, 10, 3, bootstrap=False)
					# data, epochs, minibatch_size, learning_rate, bootstrap