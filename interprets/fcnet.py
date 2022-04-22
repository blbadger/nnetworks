# fcnet.py
# import standard libraries
import string
import time
import math
import random

# import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import scipy
from sklearn.utils import shuffle
from prettytable import PrettyTable
from scipy.interpolate import make_interp_spline

import torch
import torch.nn as nn

# import local libraries
from network_interpret import StaticInterpret

# turn 'value set on df slice copy' warnings off
pd.options.mode.chained_assignment = None


class MultiLayerPerceptron(nn.Module):

	def __init__(self, input_size, output_size):

		super().__init__()
		self.input_size = input_size
		hidden1_size = 900
		hidden2_size = 100
		hidden3_size = 20
		self.input2hidden = nn.Linear(input_size, hidden1_size)
		self.hidden2hidden = nn.Linear(hidden1_size, hidden2_size)
		self.hidden2hidden2 = nn.Linear(hidden2_size, hidden3_size)
		self.hidden2output = nn.Linear(hidden3_size, output_size)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.3)

	def forward(self, input):
		"""
		Forward pass through network

		Args:
			input: torch.Tensor object of network input, size [n_letters * length]

		Return: 
			output: torch.Tensor object of size output_size

		"""

		out = self.input2hidden(input)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.hidden2hidden(out)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.hidden2hidden2(out)
		out = self.relu(out)
		out = self.dropout(out)

		output = self.hidden2output(out)
		return output


class Format:

	def __init__(self, file, training=True):

		df = pd.read_csv(file)	
		df = df.applymap(lambda x: '' if str(x).lower() == 'nan' else x)
		df = df[:20000]
		length = len(df['Elapsed Time'])
		self.input_fields = ['Store Number', 
							'Market', 
							'Order Made',
							'Cost',
							'Total Deliverers', 
							'Busy Deliverers', 
							'Total Orders',
							'Estimated Transit Time',
							'Linear Estimation']

		if training:
			df = shuffle(df)
			df.reset_index(inplace=True)

			# 80/20 training/validation split
			split_i = int(length * 0.8)

			training = df[:][:split_i]
			self.training_inputs = training[self.input_fields]
			self.training_outputs = [i for i in training['positive_two'][:]]

			validation_size = length - split_i
			validation = df[:][split_i:split_i + validation_size]
			self.validation_inputs = validation[self.input_fields]
			self.validation_outputs = [i for i in validation['positive_two'][:]]
			self.validation_inputs = self.validation_inputs.reset_index()

		else:
			self.training_inputs = self.df # not actually training, but matches name for stringify


	def stringify_input(self, index, training=True):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistant structure to inputs regardless of missing values.

		Args:
			index: int, position of input

		Returns:
			array: string: str of values in the row of interest

		"""
		
		taken_ls = [4, 1, 8, 5, 3, 3, 3, 4, 4]

		string_arr = []
		if training:
			inputs = self.training_inputs.iloc[index]
		else:
			inputs = self.validation_inputs.iloc[index]

		fields_ls = self.input_fields
		for i, field in enumerate(fields_ls):
			entry = str(inputs[field])[:taken_ls[i]]
			while len(entry) < taken_ls[i]:
				entry += '_'
			string_arr.append(entry)

		string = ''.join(string_arr)
		return string


	@classmethod
	def string_to_tensor(self, input_string):
		"""
		Convert a string into a tensor

		Args:
			string: str, input as a string

		Returns:
			tensor: torch.Tensor() object
		"""

		places_dict = {s:int(s) for s in '0123456789'}
		for i, char in enumerate('. -:_'):
			places_dict[char] = i + 10

		# vocab_size x batch_size x embedding dimension (ie input length)
		tensor_shape = (len(input_string), 1, 15) 
		tensor = torch.zeros(tensor_shape)

		for i, letter in enumerate(input_string):
			tensor[i][0][places_dict[letter]] = 1.

		tensor = tensor.flatten()
		return tensor 


	def random_sample(self):
		"""
		Choose a random index from a training set

		Args:
			None

		Returns:
			output: string
			input: string
			output_tensor: torch.Tensor
			input_tensor: torch.Tensor
		"""
		index = random.randint(0, len(self.training_inputs['store_id']) - 1)

		output = self.training_outputs['etime'][index]
		output_tensor = torch.tensor(output)

		input_string = self.stringify_inputs(index)
		input_tensor = self.string_to_tensor(input_string)

		return output, input, output_tensor, input_tensor


	def sequential_tensors(self, training=True):
		"""
		kwargs:
			training: bool

		Returns:
			input_tensors: torch.Tensor objects
			output_tensors: torch.Tensor objects
		"""

		input_tensors = []
		output_tensors = []
		if training:
			inputs = self.training_inputs
			outputs = self.training_outputs
		else:
			inputs = self.validation_inputs
			outputs = self.validation_outputs

		for i in range(len(inputs)):
			input_string = self.stringify_input(i, training=training)
			input_tensor = self.string_to_tensor(input_string)
			input_tensors.append(input_tensor)

			# convert output float to tensor directly
			output_tensors.append(torch.Tensor([outputs[i]]))

		return input_tensors, output_tensors


class ActivateNet:

	def __init__(self, epochs):
		n_letters = len('0123456789. -:_') # 15 possible characters
		file = 'data/linear_historical.csv'
		form = Format(file, training=True)
		self.input_tensors, self.output_tensors = form.sequential_tensors(training=True)
		self.validation_inputs, self.validation_outputs = form.sequential_tensors(training=False)
		self.epochs = epochs

		output_size = 1
		input_size = len(self.input_tensors[0])
		self.model = MultiLayerPerceptron(input_size, output_size)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		self.biases_arr = [[], [], []]

	@staticmethod
	def count_parameters(model):
		"""
		Display the tunable parameters in the model of interest

		Args:
			model: torch.nn object

		Returns:
			total_params: the number of model parameters

		"""

		table = PrettyTable(['Modules', 'Parameters'])
		total_params = 0
		for name, parameter in model.named_parameters():
			if not parameter.requires_grad:
				continue
			param = parameter.numel()
			table.add_row([name, param])
			total_params += param 

		print (table)
		print (f'Total trainable parameters: {total_params}')
		return total_params


	def weighted_mseloss(self, output, target):
		"""
		We are told that the true cost of underestimation is twice
		that of overestimation, so MSEloss is customized accordingly.

		Args:
			output: torch.tensor
			target: torch.tensor

		Returns:
			loss: float

		"""
		if output < target:
			loss = torch.mean((2*(output - target))**2)
		else:
			loss = torch.mean((output - target)**2)

		return loss


	def weighted_l1loss(self, output, target):
		"""
		Assigned double the weight to underestimation with L1 cost

		Args:
			output: torch.tensor
			target: torch.tensor

		Returns:
			loss: float
		"""

		if output < target:
			loss = abs(2 * (output - target))

		else:
			loss = abs(output - target)

		return loss


	def train_minibatch(self, input_tensor, output_tensor, minibatch_size):
		"""
		Train a single minibatch

		Args:
			input_tensor: torch.Tensor object 
			output_tensor: torch.Tensor object
			optimizer: torch.optim object
			minibatch_size: int, number of examples per minibatch
			model: torch.nn

		Returns:
			output: torch.Tensor of model predictions
			loss.item(): float of loss for that minibatch

		"""
		# self.model.train()
		output = self.model(input_tensor)
		output_tensor = output_tensor.reshape(minibatch_size, 1)
		loss_function = torch.nn.L1Loss()
		loss = loss_function(output, output_tensor)

		self.optimizer.zero_grad() # prevents gradients from adding between minibatches
		loss.backward()

		nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
		self.optimizer.step()

		return output, loss.item()


	def plot_predictions(self, epoch_number):
		"""
		Plot

		"""
		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		model_outputs = []

		with torch.no_grad():
			total_error = 0
			for i in range(len(self.validation_inputs)):
				input_tensor = self.validation_inputs[i]
				output_tensor = self.validation_outputs[i]
				model_output = self.model(input_tensor)
				model_outputs.append(float(model_output))

		plt.scatter([float(i) for i in self.validation_outputs], model_outputs, s=1.5)
		_, _, r2, _, _ = scipy.stats.linregress([float(i) for i in self.validation_outputs], model_outputs)
		print (f'R2 value: {r2}')
		plt.axis([0, 1600, -100, 1600]) # x-axis range followed by y-axis range
		# plt.show()
		plt.tight_layout()
		plt.savefig('regression{0:04d}.png'.format(epoch_number), dpi=400)
		plt.close()

		return


	def plot_biases(self, index):
		"""
		Image model biases as a scatterplot

		Args:
			index: int

		Returns:
			None
		"""
		self.model.eval()
		arr = self.model.hidden2hidden.bias[:6].detach().numpy()
		self.biases_arr[0].append([arr[0], arr[1]])
		self.biases_arr[1].append([arr[2], arr[3]])
		self.biases_arr[2].append([arr[4], arr[5]])
		plt.style.use('dark_background')
		plt.plot([i[0] for i in self.biases_arr[0]], [i[1] for i in self.biases_arr[0]], '^', color='white', alpha=0.7, markersize=0.1)
		plt.plot([i[0] for i in self.biases_arr[1]], [i[1] for i in self.biases_arr[1]], '^', color='red', alpha=0.7, markersize=0.1)
		plt.plot([i[0] for i in self.biases_arr[2]], [i[1] for i in self.biases_arr[2]], '^', color='blue', alpha=0.7, markersize=0.1)
		plt.axis('on')
		plt.savefig('Biases_{0:04d}.png'.format(index), dpi=400)
		plt.close()

		return

	def heatmap_weights(self, index):
		"""
		Plot model weights of one layer as a heatmap

		Args:
			index: int

		Returns:
			None

		"""
		self.model.eval()
		arr = self.model.hidden2hidden2.weight.detach()
		arr = torch.reshape(arr, (2000, 1))
		arr = arr[:44*44]
		arr = torch.reshape(arr, (44, 44))
		arr = arr.numpy()
		plt.imshow(arr, interpolation='spline16', aspect='auto', cmap='inferno')
		plt.style.use('dark_background')
		plt.axis('off')
		plt.savefig('heatmap_{0:04d}.png'.format(index), dpi=400, bbox_inches='tight', pad_inches=0)
		plt.close()
		return


	def heatmap_biases(self, index):
		"""
		Plot model biases as a rectangular heatmap

		Args:
			index: int

		Returns:
			None

		"""
		self.model.eval()
		arr = self.model.hidden2hidden.bias.detach().numpy()

		# convert to 10x10 array
		arr2 = []
		j = 0
		while j in range(len(arr)):
			ls = []
			while len(ls) < 10 and j < len(arr):
				ls.append(round(arr[j], 3))
				j += 1
			arr2.append(ls)
		
		arr2 = np.array(arr2)
		plt.imshow(arr2, interpolation='spline16', aspect='auto', cmap='inferno')
		for (y, x), label in np.ndenumerate(arr2):
			plt.text(x, y, '{:1.3f}'.format(label), ha='center', va='center')

		plt.axis('off')
		plt.rcParams.update({'font.size': 6})
		# plt.style.use('dark_background')
		plt.savefig('heatmap_{0:04d}.png'.format(index), dpi=400, bbox_inches='tight', pad_inches=0)
		plt.close()

		return


	def train_model(self, minibatch_size=32):
		"""
		Train the neural network.

		Args:
			model: MultiLayerPerceptron object
			optimizer: torch.optim object
			minibatch_size: int

		Returns:
			None

		"""

		self.model.train()
		epochs = self.epochs
		count = 0
		for epoch in range(epochs):
			pairs = [[i, j] for i, j in zip(self.input_tensors, self.output_tensors)]
			random.shuffle(pairs)
			input_tensors = [i[0] for i in pairs]
			output_tensors = [i[1] for i in pairs]
			total_loss = 0

			for i in range(0, len(input_tensors) - minibatch_size, minibatch_size):
				print (count)
				input_batch = torch.stack(input_tensors[i:i + minibatch_size])
				output_batch = torch.stack(output_tensors[i:i + minibatch_size])

				# skip the last batch if it is smaller than the other batches
				if len(input_batch) < minibatch_size:
					break

				output, loss = self.train_minibatch(input_batch, output_batch, minibatch_size)
				total_loss += loss
				if count % 25 == 0:
					# interpret = StaticInterpret(self.model, self.validation_inputs, self.validation_outputs)
					self.plot_predictions(count//25)
					# interpret.heatmap(count, method='combined')
				count += 1

			print (f'Epoch {epoch} complete: {total_loss} loss')
			# self.test_model()

		return


	def test_model(self):
		"""
		Test the model using a validation set

		Args:
			None

		Returns:
			None

		"""

		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		loss = torch.nn.L1Loss()

		with torch.no_grad():
			total_error = 0
			for i in range(len(self.validation_inputs)):
				input_tensor = self.validation_inputs[i]
				output_tensor = self.validation_outputs[i]
				model_output = self.model(input_tensor)
				total_error += loss(model_output, output_tensor).item()

		print (f'Test Average Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')

		with torch.no_grad():
			total_error = 0
			for i in range(len(self.input_tensors[:2000])):
				input_tensor = self.input_tensors[i]
				output_tensor = self.output_tensors[i]
				model_output = self.model(input_tensor)
				total_error += loss(model_output, output_tensor).item()

		print (f'Training Average Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')
		return


	def predict(self, model, test_inputs):
		"""
		Make predictions with a model.

		Args:
			model: Transformer() object
			test_inputs: torch.tensor inputs of prediction desired

		Returns:
			prediction_array: arr[int] of model predictions

		"""
		model.eval()
		prediction_array = []

		with torch.no_grad():
			for i in range(len(test_inputs['index'])):
				prediction_array.append(model_output)

		return prediction_array


network = ActivateNet(200)
network.train_model()
network.test_model()

input_tensors, output_tensors = network.validation_inputs, network.validation_outputs
print (input_tensors[0].shape)
model = network.model
interpret = StaticInterpret(model, input_tensors, output_tensors)
interpret.readable_interpretation(1, method='average')
interpret.heatmap(1)

















