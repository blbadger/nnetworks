# transformer_regressor.py
# Transformer-based neural network for regression and

# import standard libraries
import string
import time
import math
import random

# import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# turn 'value set on df slice copy' warnings off
pd.options.mode.chained_assignment = None


class MultiLayerPerceptron(nn.Module):

	def __init__(self, input_size, output_size):

		super().__init__()
		self.input_size = input_size
		hidden1_size = 500
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


class Format():

	def __init__(self, file, training=True):

		df = pd.read_csv(file)	
		df = df.applymap(lambda x: '' if str(x).lower()[0] == 'n' else x)
		df = df[:10000]
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
			self.training_outputs = [i for i in training['Elapsed Time'][:]]

			validation_size = length - split_i
			validation = df[:][split_i:split_i + validation_size]
			self.val_inputs = validation[self.input_fields]
			self.val_outputs = [i for i in validation['Elapsed Time'][:]]

		else:
			self.training_inputs = self.df # not actually training, but matches name for stringify



	def stringify_inputs(self, index):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistant structure to inputs regardless of missing values.

		Args:
			index: int, position of input

		Returns:
			array: string: str of values in the row of interest

		"""


		taken_ls = [4, 1, 8, 4, 3, 3, 3, 4, 4]
		i = index

		string_arr = []
		fields_ls = self.input_fields
		for i, field in enumerate(fields_ls):
			entry = str(self.training_inputs[field][i])[:taken_ls[i]]
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

		tensor = tensor.flatten()
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


	def sequential_tensors(self):
		"""
		Choose one example from the test set, such that repeated
		choices iterate over the entire set.

		"""

		input_tensors = []
		output_tensors = []
		for i in range(len(self.training_inputs)):
			input_string = self.stringify_inputs(i)
			input_tensor = self.string_to_tensor(input_string)
			input_tensors.append(input_tensor)

			# convert output float to tensor directly
			output_tensors.append(torch.Tensor([self.training_outputs[i]]))

		return input_tensors, output_tensors


def weighted_mseloss(output, target):
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


def weighted_l1loss(output, target):
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


def train_minibatch(input_tensor, output_tensor, optimizer, minibatch_size, model):
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

	output = model(input_tensor)
	output_tensor = output_tensor.reshape(minibatch_size, 1)
	loss_function = torch.nn.L1Loss()

	loss = loss_function(output, output_tensor)
	optimizer.zero_grad() # prevents gradients from adding between minibatches
	loss.backward()

	nn.utils.clip_grad_norm_(model.parameters(), 0.3)
	optimizer.step()

	return output, loss.item()


def train_model(model, optimizer, input_tensors, output_tensors, minibatch_size=16):
	"""
	Train the mlp model

	Args:
		model: MultiLayerPerceptron object
		optimizer: torch.optim object

	kwargs:
		minibatch_size: int

	Returns:
		None

	"""

	model.train()
	epochs = 50

	for epoch in range(epochs):
		pairs = [[i, j] for i, j in zip(input_tensors, output_tensors)]
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

			output, loss = train_minibatch(input_batch, output_batch, optimizer, minibatch_size, model)
			total_loss += loss

		print (f'Epoch {epoch} complete: {total_loss} loss')

	return


def train_network(model, optimizer, file, minibatch_size=16):
	"""
	On-line training with random samples

	Args:
		model: Transformer object
		optimizer: torch.optim object of choice

	kwags:
		minibatch_size: int, number of samples per gradient update

	Return:
		none (modifies model in-place)

	"""

	model.train()
	current_loss = 0
	training_data = Format(file, training=True)

	# training iteration and epoch number specs
	n_epochs = 50

	start = time.time()
	for i in range(n_epochs):
		random.shuffle(input_samples)
		for i in range(0, len(input_samples), minibatch_size):
			if len(input_samples) - i < minibatch_size:
				break

			input_tensor = torch.cat([input_samples[i+j] for j in range(minibatch_size)])
			output_tensor = torch.cat([output_samples[i+j] for j in range(minibatch_size)])

			# define the output and backpropegate loss
			output, loss = train_random_input(output_tensor, input_tensor, optimizer)

			# sum to make total loss
			current_loss += loss 

			if i % n_per_epoch == 0 and i > 0:
				etime = time.time() - start
				ave_error = round(current_loss / n_per_epoch, 2)
				print (f'Epoch {i//n_per_epoch} complete \n Average error: {ave_error} \n Elapsed time: {round(etime, 2)}s \n' + '~'*30)
				current_loss = 0

	return


# TODO: switch possible characters to ascii
n_letters = len('0123456789. -:')
file = 'data/linear_historical.csv'
f = Format(file, training=True)
input_tensors, output_tensors = f.sequential_tensors()
print (len(input_tensors))
output_size = 1
input_size = len(input_tensors[0])
model = MultiLayerPerceptron(input_size, output_size)
loss_function = nn.CrossEntropyLoss() # integer labels expected
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_model(model, optimizer, input_tensors, output_tensors)


def test_network(model, validation_inputs, validation_outputs):
	"""

	"""

	model.eval() # switch to evaluation mode (silence dropouts etc.)
	count = 0

	with torch.no_grad():
		total_error = 0
		for i in range(len(validation_inputs)):
			output, input, output_tensor, input_tensor = Format.sequential_sample(dataframe, i)
			model_output = model(input_tensor)
			total_error += weighted_mseloss(model_output, output_tensor)

	print (f'Average Error: {round(total_error / len(validation_inputs), 2)}')
	return


def predict(model, test_inputs):
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












