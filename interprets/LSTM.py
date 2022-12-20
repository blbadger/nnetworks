# transformer_regressor.py
# Transformer-based neural network for regression and

# import standard libraries
import string
import time
import math
import random

# import third-party libraries
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
import sklearn
from sklearn.utils import shuffle
import scipy

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# turn 'value set on df slice copy' warnings off
pd.options.mode.chained_assignment = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)

class LSTM(nn.Module):

	def __init__(self, n_letters, output_size, input_size, hidden_size, nlayers=1):

		super().__init__()
		self.output_size = output_size
		self.n_letters = n_letters
		self.num_layers = nlayers
		self.hidden_size = hidden_size

		self.lstm = nn.LSTM(n_letters, hidden_size, nlayers, dropout=0., batch_first=False)
		self.hidden2output = nn.Linear(hidden_size, output_size)

	def forward(self, input):
		output, hidden = self.lstm(input)
		output = self.hidden2output(output[-1])
		return output


class Format():

	def __init__(self, file, training=True):

		df = pd.read_csv(file)	
		df = df.applymap(lambda x: '' if str(x).lower() == 'nan' else x)
		df = df[:1000]
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

			# 80/20 training/test split
			split_i = int(length * 0.8)

			training = df[:][:split_i]
			self.training_inputs = training[self.input_fields]
			self.training_outputs = [i for i in training['positive_control'][:]]

			validation_size = length - split_i
			validation = df[:][split_i:split_i + validation_size]
			self.validation_inputs = validation[self.input_fields]
			self.validation_outputs = [i for i in validation['positive_control'][:]]
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


	def unstructured_stringify(self, index, training=True, pad=True, length=50):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistant structure to inputs regardless of missing values.

		Args:
			index: int, position of input

		Returns:
			array: string: str of values in the row of interest

		"""

		string_arr = []
		if training:
			inputs = self.training_inputs.iloc[index]
		else:
			inputs = self.validation_inputs.iloc[index]

		fields_ls = self.input_fields
		for i, field in enumerate(fields_ls):
			entry = str(inputs[field])
			string_arr.append(entry)


		string = ''.join(string_arr)
		if pad:
			if len(string) < length:
				string += '_' * (length - len(string))
			if len(string) > length:
				string = string[:length]

		return string



	@classmethod
	def string_to_tensor(self, input_string):
		"""
		Convert a string into a tensor

		Args:
			string: str, input as a string

		Returns:
			tensor
		"""

		# TODO: switch to ASCII (upper and lowercase and a few special chars)
		places_dict = {s:int(s) for s in '0123456789'}
		places_dict['.'] = 10
		places_dict[' '] = 11
		places_dict['-'] = 12
		places_dict[':'] = 13
		places_dict['_'] = 14

		# vocab_size x batch_size x embedding dimension (ie input length)
		tensor_shape = (len(input_string), 1, 15) 
		tensor = torch.zeros(tensor_shape)

		for i, letter in enumerate(input_string):
			tensor[i][0][places_dict[letter]] = 1.

		# tensor = tensor.flatten()
		return tensor 


	def random_sample(self):
		"""
		Choose a random index from a training set
		
		"""
		index = random.randint(0, len(self.training_inputs['store_id']) - 1)

		output = self.training_outputs['etime'][index]
		output_tensor = torch.tensor(output)

		input_string = self.stringify_inputs(index)
		input_tensor = self.string_to_tensor(input_string)

		return output, input, output_tensor, input_tensor


	def sequential_tensors(self, training=True):
		"""
		
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
			input_string = self.unstructured_stringify(i, training=training)
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
		self.validation_inputs, self.validation_outputs = form.sequential_tensors(training=True)
		self.epochs = epochs

		output_size = 1
		input_size = len(self.input_tensors[0])
		self.model = LSTM(n_letters, output_size, input_size, 1000)
		self.model.to(device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)



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
		output = self.model(input_tensor.to(device))
		output_tensor = output_tensor.reshape(minibatch_size, 1).to(device)
		loss_function = torch.nn.L1Loss()
		loss = loss_function(output, output_tensor)

		self.optimizer.zero_grad() # prevents gradients from adding between minibatches
		loss.backward()
		self.optimizer.step()

		return output, loss.item()


	def plot_predictions(self, epoch_number):
		"""

		"""
		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		loss = torch.nn.L1Loss()
		model_outputs = []

		with torch.no_grad():
			total_error = 0
			for i in range(len(self.validation_inputs)):
				input_tensor = self.validation_inputs[i].reshape(50, 1, 15).to(device)
				output_tensor = self.validation_outputs[i].to(device)
				model_output = self.model(input_tensor)
				model_outputs.append(float(model_output))

		plt.scatter([float(i) for i in self.validation_outputs], model_outputs, s=1.5)
		plt.axis([0, 1600, -100, 1600]) # x-axis range followed by y-axis range
		plt.xlabel('')
		# plt.show()
		plt.tight_layout()
		plt.savefig('regression{0:04d}.png'.format(epoch_number), dpi=400)
		plt.close()
		return


	def train_model(self, minibatch_size=128):
		"""
		Train the mlp model

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
			print (epoch)
			pairs = [[i, j] for i, j in zip(self.input_tensors, self.output_tensors)]
			random.shuffle(pairs)
			input_tensors = [i[0] for i in pairs]
			output_tensors = [i[1] for i in pairs]
			total_loss = 0

			for i in range(0, len(input_tensors) - minibatch_size, minibatch_size):

				# stack tensors to make shape (minibatch_size, input_size)
				input_batch = torch.stack(input_tensors[i:i + minibatch_size])
				output_batch = torch.stack(output_tensors[i:i + minibatch_size])
				# skip the last batch if too small
				if len(input_batch) < minibatch_size:
					break
				input_batch = input_batch.reshape(50, minibatch_size, 15)

				output, loss = self.train_minibatch(input_batch, output_batch, minibatch_size)
				total_loss += loss
			print (f'Loss: {total_loss}')

		return


	def test_model(self):
		"""

		"""

		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		loss = torch.nn.L1Loss()

		model_outputs, true_outputs = [], []
		with torch.no_grad():
			total_error = 0
			for i in range(len(self.validation_inputs)):
				input_tensor = self.validation_inputs[i].reshape(50, 1, 15).to(device)
				output_tensor = self.validation_outputs[i].reshape(1).to(device)
				model_output = self.model(input_tensor)
				total_error += loss(model_output, output_tensor).item()
				model_outputs.append(float(model_output))
				true_outputs.append(float(output_tensor))

		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(model_outputs, true_outputs)
		print (f'Mean Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')
		print (f'R2 value: {r_value}')
		return




epochs = 1000
network = ActivateNet(epochs)
network.train_model()
network.test_model()
network.plot_predictions(epochs)



