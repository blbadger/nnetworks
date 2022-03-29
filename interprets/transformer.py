# transformer.py

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
from statsmodels.formula.api import ols

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# import custom libraries
from network_interpret import Interpret 
from data_formatter import Format

# Turn 'value set on df slice copy' warnings off, but
# note that care should be taken to match pandas dataframe
# column to the appropriate type
pd.options.mode.chained_assignment = None


class Transformer(nn.Module):
	"""
	Encoder-only tranformer architecture for regression.  The approach is 
	to average across the states yielded by the transformer encoder before
	passing this to a single hidden fully connected linear layer.
	"""
	def __init__(self, output_size, n_letters, d_model, nhead, feedforward_size, nlayers, minibatch_size, dropout=0.3):

		super().__init__()
		self.posencoder = PositionalEncoding(d_model, dropout)

		encoder_layers = TransformerEncoderLayer(d_model, nhead, feedforward_size, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

		self.encoder = nn.Embedding(n_letters, d_model)
		self.d_model = d_model
		self.init_weights()
		self.transformer2hidden = nn.Linear(n_letters * 54, 50)
		self.hidden2output = nn.Linear(50, 1)
		self.relu = nn.ReLU()
		self.n_letters = n_letters
		self.minibatch_size = minibatch_size


	def init_weights(self):
		"""
		Uniform distribution weight initialization for transformer encoder

		Args:
			None

		Returns:
			None (initializes self.encoder.weight in-place)

		"""

		initrange = 0.2

		# in-place assignment of initialization to uniform distribution with range 2*initrange
		self.encoder.weight.data.uniform_(-initrange, initrange)
		return


	def forward(self, input_tensor):
		"""
		Forward pass through network

		Args:
			input_tensor: torch.Tensor of character inputs

		Returns: 
			output: torch.Tensor, linear output
		"""

		# reshape input: sequence size x batch size x embedding dimension
		length = len(input_tensor)
		input = input_tensor.reshape(length, self.minibatch_size, self.n_letters)

		# apply (relative) positional encoding
		input = self.posencoder(input)
		output = self.transformer_encoder(input)
		output = output.reshape(self.minibatch_size, length, self.n_letters)
		output = torch.flatten(output, start_dim=1)
		output = self.transformer2hidden(output)
		output = self.relu(output)
		output = self.hidden2output(output)

		# return linear-activation output
		return output


class PositionalEncoding(nn.Module):
	"""
	Encodes relative positional information on the input
	"""

	def __init__(self, model_size, dropout=0.3, max_len = 1000):

		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.model_size = model_size

		if self.model_size % 2 == 0:
			arr = torch.zeros(max_len, model_size)

		else:
			arr = torch.zeros(max_len, model_size + 1)

		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, model_size, 2).float() * (-math.log(10*max_len)/ model_size))
		arr[:, 0::2] = torch.sin(position * div_term)
		arr[:, 1::2] = torch.cos(position * div_term)

		arr = arr.unsqueeze(0).transpose(0, 1)
		self.arr = arr


	def forward(self, tensor):
		"""
		Apply positional information to input

		Args:
			tensor: torch.Tensor, network input

		Returns:
			dout: torch.Tensor of modified input

		"""

		if self.model_size % 2 == 0:
			tensor = tensor + self.arr[:tensor.size(0), :, :]

		else:
			tensor = tensor + self.arr[:tensor.size(0), :, :-1] # decrement to prevent oob

		dout = self.dropout(tensor)
		return dout


def mse_loss(output, target):
	"""
	Mean squared error loss.

	Args:
		output: torch.tensor
		target: torch.tensor

	Returns:
		loss: float

	"""

	loss = torch.mean((output - target)**2)

	return loss



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


def init_transformer(n_letters, minibatch_size):
	"""
	Initialize a transformer model

	Args:
		n_letters: int, number of ascii inputs
		emsize: int, the embedding dimension

	Returns:
		model: Transformer object

	"""

	# note that nhead (number of multi-head attention units) must be able to divide d_model
	feedforward_size = 280
	nlayers = 3
	nhead = 5
	d_model = n_letters
	n_output = 1

	model = Transformer(n_output, n_letters, d_model, nhead, feedforward_size, nlayers, minibatch_size)
	return model


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

	loss = loss_function(output, output_tensor)
	optimizer.zero_grad() # prevents gradients from adding between minibatches
	loss.backward()

	nn.utils.clip_grad_norm_(model.parameters(), 0.3)
	optimizer.step()

	return output, loss.item()

# TODO: switch possible characters to ascii
n_letters = len('0123456789. -:_')
file = 'data/linear_historical.csv'
input_tensors = Format(file, 'Elapsed Time')

input_arr, output_arr = input_tensors.transform_to_tensors()
line_length = len(input_arr[0]) // n_letters
minibatch_size = 32
model = init_transformer(n_letters, minibatch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
file = 'linear_historical.csv'
loss_function = nn.L1Loss()


def train_model(model, input_tensors, output_tensors, optimizer, minibatch_size):
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

		for i in range(0, len(pairs) - minibatch_size, minibatch_size):
			# stack tensors to make shape (minibatch_size, input_size)
			input_batch = torch.stack(input_tensors[i:i + minibatch_size])
			output_batch = torch.stack(output_tensors[i:i + minibatch_size])

			# skip the last batch if too small
			if len(input_batch) < minibatch_size:
				break

			# default tensor size sequence_len x batch_size x embedding_size
			input_batch = input_batch.reshape(line_length, minibatch_size, n_letters)
			output, loss = train_minibatch(input_batch, output_batch, optimizer, minibatch_size, model)
			total_loss += loss

		print (f'Epoch {epoch} complete: {total_loss} loss')

	return


def save_model(model):
	"""
	Saves a Transformer object state dictionary

	Args:
		model: Transformer class object

	Returns:
		None

	"""

	file_name = 'transformer.pth'
	torch.save(model.state_dict(), file_name)
	return


def evaluate_network(model):
	"""
	Evaluate network on validation data.

	Args:
		model: Transformer class object

	Returns:
		None (prints validation accuracies)

	"""

	model.eval() # switch to evaluation mode (silence dropouts etc.)
	count = 0
	validation_data = Format(file, training=True)

	with torch.no_grad():
		total_error = 0
		mae_error = 0
		weighted_mae = 0
		for i in range(len(validation_data.val_inputs)): 
			output, input, output_tensor, input_tensor = validation_data.validation()
			model_output = model(input_tensor)
			linear_output = validation_data.val_inputs['linear_ests'][i]
			if linear_output:
				model_output = (float(model_output) + linear_output) / 2

			if i < 100:
				print (model_output)

			total_error += (float(model_output) - float(output_tensor))**2
			mae_error += abs(float(model_output) - float(output_tensor))
			count += 1

			if float(model_output) < float(output_tensor):
				weighted_mae += 2*abs(float(model_output) - float(output_tensor))
			else:
				weighted_mae += abs(float(model_output) - float(output_tensor))

	rms_error = (total_error / count) ** 0.5
	print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	print (f'Validation RMS error: {round(rms_error, 2)} \n')
	print (f'Validation Weighted MAE: {weighted_mae / count}')
	print (f'Validation Mean Absolute Error: {mae_error / count}')

	return


train_model(model, input_arr, output_arr, optimizer, minibatch_size)

# model.load_state_dict(torch.load('transformer.pth'))

# evaluate_network(model)


def predict(model, file):
	"""
	Use a trained transformer (with the linear model) to predict durations

	Args:
		model: Transformer() object
		file: string

	Returns: 
		prediction_array, arr[float] of predicted durations

	"""

	model.eval()
	prediction_array = []
	test_data = Format(file, training=False)
	test_inputs = test_data.generate_test_inputs()

	with torch.no_grad():
		for i, input_tensor in enumerate(test_data.generate_test_inputs()):

			output = model(input_tensor)
			linear_output = test_data.test_inputs['linear_ests'][i]

			# average transformer output with linear model output, if it exists
			if linear_output:
				output = (float(output) + linear_output) / 2

			prediction_array.append(float(output))

	return prediction_array


def add_predictions():
	"""
	Save predictions to a csv file.

	Args:
		None

	Returns:
		None (saves df in-place)

	"""

	file = 'data_to_predict.csv'

	predictions = predict(model, file)
	df = pd.read_csv('data_to_predict.csv')
	df['predicted_duration'] = predictions
	df.to_csv('data_to_predict.csv')


# add_predictions()


model_interpretation = Interpret(model, file)
model_interpretation.graph_attributions()






