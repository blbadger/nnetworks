#! fcnetwork_scratch.py
# A fully connected feed-forward neural network from scratch 
# (no numpy, only core libraries).  Stochastic gradient descent 
# with a quadratic network cost function and sigmoid neuron
# activation functions, and on-line learning (minibatches of size 1).

### Nota Bene: this network is slow relative to fcnetwork.py, as
### all calculations are performed element-wise rather than matrix-
### wise as is the case for numpy (which arranges elements in 
### contiguous memory blocks and performs operations on the ensemble).
### Training and test sets are reduced in number because of this.
import random

class FullNetworkScratch:

	def __init__(self, architecture, activation_function):

		# initialize biases
		self.biases = []
		for layer in architecture[1:]:
			temp = []
			for i in range(layer):
				# initial biases 
				temp.append(random.normalvariate(0, 1)) 
			self.biases.append(temp)

		# initialize weights
		self.weights = []
		for i in range(1, len(architecture)):
			layer = []
			for j in range(architecture[i-1]):
				temp = []
				for k in range(architecture[i]):
					# initial weights 
					temp.append(random.normalvariate(0, 1)) 
				layer.append(temp)
			self.weights.append(layer)

		# misc. init
		self.arc = architecture
		self.activation = activation_function
		self.item_length = len(architecture)


	def activation_function(self, z):
		# Activation function for a single element
		if self.activation == 'relu':
			return max(0, z)

		if self.activation == 'sigmoid':
			return 1 / (1 + 2.71828**(-1*z))


	def activation_prime(self, z):
		# derivative of activation function for a single element
		if self.activation == 'relu':
			if z > 0:
				z = 1
			else:
				z = 0
			return z

		if self.activation == 'sigmoid':
			return self.activation_function(z)*(1 - self.activation_function(z))


	def network_output(self, input_ls):
		# Feed-forward network output
		z_matrix = []
		input_ls = [i[0] for i in input_ls]

		for index in range(self.item_length-1):
			weight, bias = self.weights[index], self.biases[index]
			total_ls = [0 for i in range(len(bias))]
			for j, w_vec in enumerate(weight):
				for i in range(len(w_vec)):
					total_ls[i] += w_vec[i] * input_ls[j]

			z_matrix.append(total_ls)
			input_ls = [self.activation_function(i) for i in total_ls]

		
		output = [self.activation_function(i) for i in z_matrix[-1]]
		return output


	def evaluate(self, test_data):
		total = 0
		for data in test_data:
			x, y = data
			output = self.network_output(x)
			index = 0
			for i in range(len(output)):
				if output[i] > output[index]:
					index = i

			index2 = 0
			for i in range(len(y)):
				if y[i] > y[index2]:
					index2 = i

			if index == index2:
				total += 1

		return total


	def cost_function_derivative(self, output_activations, y):
		# cost function derivate for individual node
		return output_activations - y


	def gradient_descent(self, data, epochs, learning_rate, bootstrap=False):
		# Trains neural network using stochastic gradient descent 
		# with optional bootstrapping.
		data = list(data)

		for epoch in range(epochs):
			for entry in data[:200]:
				self.update_network(entry, learning_rate)

			number_correct = self.evaluate(data[:200])
			number_total = len(data[:200])
			print ('Epoch {} complete: {} / {}'.format(epoch, number_correct, number_total))


	def update_network(self, entry, learning_rate):
		_input, classification = entry

		# set input activation
		activation = [i[0] for i in _input] # convert to list
		activations = [activation]
		z_vectors = []

		# compute subsequent layer activations and z vectors
		for i in range(1, self.item_length):
			z_vector = []

			for j in range(len(self.weights[i-1][0])):
				total = 0
				for k in range(len(activation)):
					total += self.weights[i-1][k][j] * activation[k]
				z_vector.append(total + self.biases[i-1][j])

			new_activation = []

			for i in range(len(z_vector)):
				new_activation.append(self.activation_function(z_vector[i]))
			activations.append(new_activation)
			activation = new_activation
			z_vectors.append(z_vector)

		self.backpropegate_error(classification, activations, z_vectors, learning_rate)

	def backpropegate_error(self, classification, activations, z_vectors, learning_rate):
		# compute output error
		classification = [i[0] for i in classification]

		error = []
		for i in range(len(z_vectors[-1])):
			diff = self.cost_function_derivative(activations[-1][i], classification[i])
			val = diff * self.activation_prime(z_vectors[-1][i])
			error.append(val)

		# Find partial derivatives of the last layer wrt error
		dc_db = [error]
		dc_dw = []
		temp_dw = []
		for err in error:
			temp = []
			for act in activations[-2]:
				temp.append(err * act)
			temp_dw.append(temp)
		dc_dw.append(temp_dw)

		# backpropegate to previous layers
		for i in range(2, self.item_length):
			dot_product = []
			for j in range(len(self.weights[-i+1])):
				total = 0
				for k in range(len(self.weights[-i+1][j])):
					total += self.weights[-i+1][j][k] * error[k]
				dot_product.append(total)

			activation_primes = []
			for j in range(len(z_vectors[-i])):
				activation_primes.append(self.activation_prime(z_vectors[-i][j]))

			error = [x*y for x, y in zip(dot_product, activation_primes)]

			# update partial bias derivatives 
			dc_db.append(error)
			
			# update partial weights derivatives (same as above)
			temp_dw = []
			for err in error:
				temp = []
				for act in activations[-i-1]:
					temp.append(err * act)
				temp_dw.append(temp)
			dc_dw.append(temp_dw)

		# revese gradient weights and biases to match self.weights/biases
		dc_db.reverse()
		dc_dw.reverse()

		# update weights and biases via gradient descent (move the 
		# opposite direction of the gradient)
		lr = learning_rate

		for i in range(len(self.biases)):
			for j in range(len(self.biases[i])):
				self.biases[i][j] = self.biases[i][j] - lr * dc_db[i][j]

		for i in range(len(self.weights)):
			for j in range(len(self.weights[i])):
				for k in range(len(self.weights[i][j])):
					self.weights[i][j][k] = self.weights[i][j][k] - lr * dc_dw[i][k][j]



### MNIST loading framework from Nielsen
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = FullNetworkScratch([784, 20, 10], activation_function='sigmoid')

net.gradient_descent(training_data, 50, 0.1) # data, epochs, learning_rate

