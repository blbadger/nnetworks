#! fcnetwork_scratch.py
# A fully connected feed-forward neural network from scratch 
# (no numpy, only core libraries).  Stochastic gradient descent 
# cost function minimization, the option of ReLu or sigmoid 
# activation functions, trained on individual training examples

### TODO: network only learns with sigmoid activation function: fix
### relu design to test this function as well.
import random

class FullNetworkScratch:

	def __init__(self, architecture, activation_function):

		# initialize biases
		self.biases = []
		for layer in architecture[1:]:
			temp = []
			for i in range(layer):
				# initial biases 
				temp.append(random.gauss(0, 1)) 
			self.biases.append(temp)

		# initialize weights
		self.weights = []
		for i in range(1, len(architecture)):
			layer = []
			for j in range(architecture[i]):
				temp = []
				for k in range(architecture[i-1]):
					# initial weights 
					temp.append(random.gauss(0, 1)) 
				layer.append(temp)
			self.weights.append(layer)

		# misc. init
		self.arc = architecture
		self.activation = activation_function
		self.item_length = len(architecture)


	def activation_function(self, z):
		# Activation function for a single element
		if self.activation == 'relu':
			for i in range(len(z)):
				for j in range(len(z[i])):
					z[i][j] = max(0, z[i][j])
			return z

		if self.activation == 'sigmoid':
			return 1 / (1 + 2.71828**(-1*z))


	def activation_prime(self, z):
		# derivative of activation function for a single element
		if self.activation == 'relu':
			for i in range(len(z)):
				if z[i] > 0:
					z[i] = 1
			return z

		if self.activation == 'sigmoid':
			return self.activation_function(z)*(1 - self.activation_function(z))


	def network_output(self, output):
		# Feed-forward network output
		for index in range(self.item_length-1):
			weight, bias = self.weights[index], self.biases[index]
			z_matrix = []
			total = 0
			for w, o in zip(weight, output):
				total += w * o
			z_matrix.append(total + bias)
		output = self.activation_function(z_matrix)
		return output


	def evaluate(self, test_data):
		total = 0
		for data in test_data:
			x, y = data
			outputs = self.network_output(x)
			for output in outputs:
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
		# works on lists
		return output_activations - y


	def gradient_descent(self, data, epochs, learning_rate, bootstrap=False):
		# Trains neural network using stochastic gradient descent 
		# with optional bootstrapping.
		data = list(data)
		for epoch in range(epochs):
			for entry in data:
				self.update_network(entry, learning_rate)

			number_correct = self.evaluate(data[:10000])
			number_total = len(data[:10000])
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
			for j in range(len(self.weights[i-1])):
				total = 0
				for k in range(len(self.weights[i-1][j])):
					total += self.weights[i-1][j][k] * activation[k]
				z_vector.append(total + self.biases[i-1][j])
			new_activation = []
			for i in range(len(z_vector)):
				# sigmoid function activation
				new_activation.append(1 / (1 + 2.71828**(-1*z_vector[i])))
			activations.append(new_activation)
			z_vectors.append(z_vector)

		# compute output error
		error = []
		for i in range(len(z_vectors[-1])):
			diff = self.cost_function_derivative(z_vectors[-1][i], classification[i])
			val = diff * self.activation_prime(z_vectors[-1][i])
			val = val[0]
			error.append(val)

		# Find partial derivatives of the last layer wrt error
		dc_db = [error]
		dc_dw = []

		# dc_dw.append(np.dot(error, activations[-2].transpose()))
		print (activations[-2])
		print (error)
		for i in range(len(error)):
			total = 0
			for j in range(len(error[-2][i])):
				total += error[i] * activations[-2][j]
			dc_dw.append(total)
		print (dc_dw)

		# backpropegate to previous layers
		for i in range(2, self.item_length):
			dot_product = []
			for j in range(len(self.weights[-i+1])):
				total = 0
				for k in range(len(self.weights[-i+1])):
					total += self.weights[-i+1][j][k] * error[k]
				dot_product.append(total)
			activation_primes = []

			# self.activation_prime(z_vectors[-i][j] for j in range(len(z_vectors[-i])))
			# error = [i*j for i, j in zip(dot_product, self.activation_prime(z_vectors))]

			# update partial derivatives with new error
			dc_db.append(error)
			dot_prod = []
			for i in range(len(error)):
				print (error[i])
			dc_dw.append(np.dot(error, activations[-i-1].transpose()))

		# update weights and biases
		dc_db.reverse()
		dc_dw.reverse()

		partial_db = [dnb for dnb in dc_db]
		partial_dw = [dnw.transpose() for dnw in dc_dw]

		# gradient descent (move the opposite direction of the gradient)
		lr = learning_rate
		self.weights = [w - lr*dw for w, dw in zip(self.weights, partial_dw)]
		self.biases = [b - lr*db for b, db in zip(self.biases, partial_db)]


### MNIST loading framework from Nielsen
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


net = FullNetworkScratch([784, 20, 10], activation_function='sigmoid')

net.gradient_descent(training_data, 20, 0.1, bootstrap=False) # data, epochs, learning_rate, bootstrap
