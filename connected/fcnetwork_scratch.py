#! fcnetwork_scratch.py
# A fully connected feed-forward neural network from scratch,
# using no libraries.

###TODO: finish converting network off numpy starting points.

class FullNetwork:

	def __init__(self, architecture, activation_function):
		# initialize biases of 0.5 to start
		self.biases = []
		for layer in architecture:
			self.biases.append([0.5 for i in range(layer)])

		# initialize weights of 0.5 to start
		self.weights = []
		for i in range(1, len(architecture)):
			j = architecture[i]
			k = architecture[i-1]
			self.weights.append(np.random.randn(k, j))

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
			return self.activation_function(z)*(1 - self.activation_function(z))


	def network_output(self, output):
		# Feed-forward network output

		for index in range(self.item_length):
			weight, bias = self.weights[index], self.biases[index]
			output = self.activation_function(np.dot(weight, output) + bias)

		return output


	def evaluate(self, data):
		# Evaluate network performance on dataset

		results = [max(self.network_output(x), y) for x, y in data]
		return sum([i for i in results if int(x) == y])


	def cost_function_derivative(self, output_activations, y):
		return output_activations - y


	def gradient_descent(self, data, epochs, learning_rate, bootstrap=False):
		# Trains neural network using stochastic gradient descent 
		# with optional bootstrapping.

		data = list(data)
		for epoch in range(epochs):
			for entry in data:
				self.update_network(entry)
				break

			print ('Epoch {0:01d} complete: {0:02d} /  \
				{0:03d} '.format(epoch, self.evaluate(data), 50000))


	def update_network(self, entry):
		_input, classification = entry

		# set input activation
		activation = _input
		activations = [activation]
		z_vectors = []

		# compute subsequent layer activations and z vectors
		for i in range(1, self.item_length):
			z_vector = np.dot(self.weights[i-1].transpose(), activation) + self.biases[i-1]
			activation = self.activation_function(z_vector)
			activations.append(activation)
			z_vectors.append(z_vector)

		# compute output error
		error = self.cost_function_derivative(activations[-1], classification) \
				* self.activation_prime(z_vectors[-1])

		# Find partial derivatives of the last layer wrt error
		dc_db = []
		dc_dw = []
		dc_db.append(error)
		dc_dw.append(np.dot(error, activations[-2].transpose()))

		# backpropegate to previous layers
		for i in range(2, self.item_length):
			error = np.dot(self.weights[-i+1], error) * self.activation_prime(z_vectors[-i])

			# update partial derivatives with new error
			dc_db.append(error)
			dc_dw.append(np.dot(error, activations[-i-1].transpose()))

		# return gradient of the cost function
		dc_db.reverse()
		dc_dw.reverse()
		return dc_db, dc_dw


import mnist_loader
import fcnetwork

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = fcnetwork.FullNetwork([784, 20, 10], activation_function='sigmoid')

net.gradient_descent(training_data, 5, 3, bootstrap=False) # data, epochs, learning_rate, bootstrap








