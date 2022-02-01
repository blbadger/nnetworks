#! fcnetwork.py
# A fully connected feed-forward neural network from scratch 
# (allowing for numpy & random).  Gradient descent for quadratic
# cost function minimization, the option of ReLu or sigmoid 
# activation functions, trained on individual training examples

### TODO: network only learns with sigmoid activation function: fix
### relu design to test this function as well.

import random
import numpy as np 

# Nielsen's mnist dataset loader
import mnist_loader

class FullNetwork:

	def __init__(self, architecture, activation_function):

		# initialize biases
		self.biases = []
		for layer in architecture[1:]:
			self.biases.append(np.random.randn(layer, 1))
		self.arc = architecture

		self.activation = activation_function

		# initialize weights
		self.weights = []
		for i in range(1, len(architecture)):
			j = architecture[i]
			k = architecture[i-1]
			self.weights.append(np.random.randn(k, j))

		self.item_length = len(architecture)


	def activation_function(self, z):
		# Activation function, rectified linear unit
		z = np.array(z)
		zero_array = np.zeros(z.shape)

		if self.activation == 'relu':
			return np.maximum(zero_array, z)

		if self.activation == 'sigmoid':
			return 1 / (1 + np.exp(-z))


	def activation_prime(self, z):
		# derivative of ReLu or sigmoid expression

		if self.activation == 'relu':
			return np.where(z > 0, 0, 1)

		if self.activation == 'sigmoid':
			return self.activation_function(z)*(1 - self.activation_function(z))


	def network_output(self, output):
		# Feed-forward network output
		for index in range(self.item_length-1):
			weight, bias = self.weights[index], self.biases[index]
			output = self.activation_function(np.dot(weight.T, output) + bias)

		return output


	def evaluate(self, test_data):
		test_results = [(np.argmax(self.network_output(x)), np.argmax(y)) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)


	def cost_function_derivative(self, output_activations, y):
		# quadratic cost function derivative
		return output_activations - y


	def gradient_descent(self, data, epochs, learning_rate, bootstrap=False):
		# Trains neural network using stochastic gradient descent 
		# with optional bootstrapping.
		data = list(data)
		for epoch in range(epochs):
			for entry in data[:40000]:
				self.update_network(entry, learning_rate)
			number_correct = self.evaluate(data[40000:50000])
			number_total = len(data[40000:50000])
			print ('Epoch {} complete: {} / {}'.format(epoch, number_correct, number_total))


	def update_network(self, entry, learning_rate):
		_input, classification = entry

		# set input activation
		activation = _input
		activations = [activation]
		z_vectors = []

		# compute subsequent layer activations and z vectors
		for i in range(1, self.item_length):
			z_vector = np.dot(self.weights[i-1].T, activation) + self.biases[i-1]
			activation = self.activation_function(z_vector)
			activations.append(activation)
			z_vectors.append(z_vector)

		# compute output error
		error = self.cost_function_derivative(activations[-1], classification) * self.activation_prime(z_vectors[-1])

		# Find partial derivatives of the last layer wrt error
		dc_db = []
		dc_dw = []
		dc_db.append(error)
		dc_dw.append(np.dot(error, activations[-2].T))

		# backpropegate to previous layers
		for i in range(2, self.item_length):
			error = np.dot(self.weights[-i+1], error) * self.activation_prime(z_vectors[-i])

			# update partial derivatives with new error
			dc_db.append(error)
			dc_dw.append(np.dot(error, activations[-i-1].T))

		# update weights and biases
		dc_db.reverse()
		dc_dw.reverse()

		dc_dw = np.array(dc_dw)

		partial_db = [dnb for dnb in dc_db]
		partial_dw = [dnw.T for dnw in dc_dw]

		# gradient descent (move the opposite direction of the gradient)
		lr = learning_rate
		self.weights = [w - lr*dw for w, dw in zip(self.weights, partial_dw)]
		self.biases = [b - lr*db for b, db in zip(self.biases, partial_db)]



training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = FullNetwork([784, 20, 10], activation_function='sigmoid')

net.gradient_descent(training_data, 20, 0.1, bootstrap=False) # data, epochs, learning_rate, bootstrap

























