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
from prettytable import PrettyTable

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib

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
		self.d_model = d_model
		encoder_layers = TransformerEncoderLayer(d_model, nhead, feedforward_size, dropout, batch_first=True)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
		self.transformer2hidden = nn.Linear(n_letters * d_model, 50)
		self.hidden2output = nn.Linear(50, 1)
		self.relu = nn.ReLU()
		self.n_letters = n_letters
		self.minibatch_size = minibatch_size

	def forward(self, input_tensor):
		"""
		Forward pass through network

		Args:
			input_tensor: torch.Tensor of character inputs

		Returns: 
			output: torch.Tensor, linear output
		"""

		# apply (relative) positional encoding
		input_encoded = self.posencoder(input_tensor)
		output = self.transformer_encoder(input_encoded)

		# output shape: same as input (batch size x sequence size x embedding dimension)
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

	def __init__(self, model_size, dropout=0.3, max_len=1000):

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
			tensor = tensor + self.arr[:tensor.size(0), :, :-1] # -1 ending to prevent oob

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


def init_transformer(embedding_dim, minibatch_size, line_length):
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
	d_model = embedding_dim
	n_letters = line_length
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
	model.train()
	output = model(input_tensor)
	output_tensor = output_tensor.reshape(minibatch_size, 1)

	loss = loss_function(output, output_tensor)
	optimizer.zero_grad() # prevents gradients from adding between minibatches
	loss.backward()

	nn.utils.clip_grad_norm_(model.parameters(), 0.3)
	optimizer.step()

	return output, loss.item()

def plot_predictions(model, validation_inputs, validation_outputs, count):
	"""
	Plot

	"""
	
	model_outputs = []

	with torch.no_grad():
		total_error = 0
		for i in range(len(validation_inputs)):
			input_tensor = validation_inputs[i]
			input_tensor = input_tensor.reshape(1, len(input_tensor), len(input_tensor[0]))
			output_tensor = validation_outputs[i]
			model_output = model(input_tensor)
			model_outputs.append(float(model_output))

	plt.scatter([float(i) for i in validation_outputs], model_outputs, s=1.5)
	plt.axis([0, 1600, -100, 1600]) # x-axis range followed by y-axis range
	# plt.show()
	plt.tight_layout()
	plt.savefig('regression{0:04d}.png'.format(count), dpi=400)
	plt.close()
	return

# TODO: switch possible characters to ascii
embedding_dim = len('0123456789. -:_')
file = 'data/linear_historical.csv'
input_tensors = Format(file, 'positive_control')

input_arr, output_arr = input_tensors.transform_to_tensors(training=True, flatten=False)
test_inputs, test_outputs = input_tensors.transform_to_tensors(training=False, flatten=False)
line_length = len(input_arr[0])
minibatch_size = 32
model = init_transformer(embedding_dim, minibatch_size, line_length)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
file = 'data/linear_historical.csv'
loss_function = nn.L1Loss()

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

print (count_parameters(model))

def quiver_gradients(index, model, input_tensor, output_tensor, minibatch_size=32):
		"""
		Plot a quiver map of the gradients of a chosen layer's parameters

		Args:
			index: int, current training iteration
			model: pytorch transformer model
			input_tensor: torch.Tensor object
			output_tensor: torch.Tensor object
		kwargs:
			minibatch_size: int, size of minibatch

		Returns:
			None (saves matplotlib pyplot figure)
		"""
		model.eval()
		layer = model.transformer_encoder.layers[0]
		x, y = layer.linear1.bias[:2].detach().numpy()
		print (x, y)
		plt.style.use('dark_background')

		x_arr = np.arange(x - 0.01, x + 0.01, 0.001)
		y_arr = np.arange(y - 0.01, y + 0.01, 0.001)

		XX, YY = np.meshgrid(x_arr, y_arr)
		dx, dy = np.meshgrid(x_arr, y_arr) # copy that will be overwritten
		for i in range(len(x_arr)):
			for j in range(len(y_arr)):
				with torch.no_grad():
					layer.linear1.bias[0] = torch.nn.Parameter(torch.Tensor([x_arr[i]]))
					layer.linear1.bias[1] = torch.nn.Parameter(torch.Tensor([y_arr[j]]))
				model.transformer_encoder.layers[0] = layer
				output = model(input_tensor)
				output_tensor = output_tensor.reshape(minibatch_size, 1)
				loss_function = torch.nn.L1Loss()
				loss = loss_function(output, output_tensor)
				optimizer.zero_grad()
				loss.backward()
				layer = model.transformer_encoder.layers[0]
				dx[j][i], dy[j][i] = layer.linear1.bias.grad[:2]

		matplotlib.rcParams.update({'font.size': 8})
		color_array = 2*(np.abs(dx) + np.abs(dy))
		plt.quiver(XX, YY, dx, dy, color_array)
		plt.plot(x, y, 'o', markersize=1)
		plt.savefig('quiver_{0:04d}.png'.format(index), dpi=400)
		plt.close()
		with torch.no_grad():
			model.transformer_encoder.layers[0].linear1.bias.grad[:2] = torch.Tensor([x, y])
		return


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
	epochs = 200
	count = 0

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

			# tensor shape: batch_size x sequence_len x embedding_size
			output, loss = train_minibatch(input_batch, output_batch, optimizer, minibatch_size, model)
			total_loss += loss
			count += 1
			if count % 25 == 0: # plot every 23 epochs for minibatch size of 32
				plot_predictions(model, test_inputs, test_outputs, count//25)
				quiver_gradients(count//25, model, input_batch, output_batch)


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
	validation_data = Format(file, 'positive_control')

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

evaluate_network(model)


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
	test_data = Format(file)
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






