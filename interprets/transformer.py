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
	def __init__(self, output_size, line_length, n_letters, nhead, feedforward_size, nlayers, minibatch_size, dropout=0.3):

		super().__init__()
		self.posencoder = PositionalEncoding(n_letters)
		encoder_layers = TransformerEncoderLayer(n_letters, nhead, feedforward_size, dropout, batch_first=True)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
		self.transformer2hidden = nn.Linear(line_length * n_letters, 50)
		self.hidden2output = nn.Linear(50, 1)
		self.relu = nn.ReLU()
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
		output = self.transformer_encoder(input_tensor)

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

	def __init__(self, model_size, max_len=1000):

		super().__init__()
		self.model_size = model_size
		if self.model_size % 2 == 0:
			arr = torch.zeros(max_len, model_size)

		else:
			arr = torch.zeros(max_len, model_size + 1)

		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, model_size, 2).float() * (-math.log(10*max_len) / model_size))
		arr[:, 0::2] = torch.sin(position * div_term)
		arr[:, 1::2] = torch.cos(position * div_term)
		arr = arr.unsqueeze(0)
		self.arr = arr


	def forward(self, tensor):
		"""
		Apply positional information to input

		Args:
			tensor: torch.Tensor, network input

		Returns:
			dout: torch.Tensor of modified input

		"""
		tensor = tensor + self.arr[:, :tensor.size(1), :tensor.size(2)]
		return tensor


class ActivateNetwork:

	def __init__(self):
		embedding_dim = len('0123456789. -:_')
		file = 'data/linear_historical.csv'
		input_tensors = Format(file, 'positive_control')

		self.train_inputs, self.train_outputs = input_tensors.transform_to_tensors(training=True, flatten=False)
		self.test_inputs, self.test_outputs = input_tensors.transform_to_tensors(training=False, flatten=False)
		self.line_length = len(self.train_inputs[0])
		self.minibatch_size = 32
		self.model = self.init_transformer(embedding_dim, self.minibatch_size, self.line_length)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		self.loss_function = nn.L1Loss()
		self.epochs = 200

	def init_transformer(self, embedding_dim, minibatch_size, line_length):
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
		n_letters = embedding_dim # set the d_model to the number of letters used
		n_output = 1
		model = Transformer(n_output, line_length, n_letters, nhead, feedforward_size, nlayers, minibatch_size)
		return model


	def plot_predictions(self, validation_inputs, validation_outputs, count):
		"""
		Plots the model's predicted values (y-axis) against the true values (x-axis)

		Args:
			model: torch.nn.Transformer module
			validation_inputs: arr[torch.Tensor] 
			validations_outputs: arr[torch.Tensor]
			count: int, iteration of plot in sequence

		Returns:
			None (saves png file to disk)
		"""

		self.model.eval()
		model_outputs = []

		with torch.no_grad():
			total_error = 0
			for i in range(len(validation_inputs)):
				input_tensor = validation_inputs[i]
				input_tensor = input_tensor.reshape(1, len(input_tensor), len(input_tensor[0]))
				output_tensor = validation_outputs[i]
				model_output = self.model(input_tensor)
				model_outputs.append(float(model_output))

		plt.scatter([float(i) for i in validation_outputs], model_outputs, s=1.5)
		plt.axis([0, 1600, -100, 1600]) # x-axis range followed by y-axis range
		# plt.show()
		plt.tight_layout()
		plt.savefig('regression{0:04d}.png'.format(count), dpi=400)
		plt.close()
		return


	def count_parameters(self):
		"""
		Display the tunable parameters in the model of interest

		Args:
			model: torch.nn object

		Returns:
			total_params: the number of model parameters

		"""

		table = PrettyTable(['Modules', 'Parameters'])
		total_params = 0
		for name, parameter in self.model.named_parameters():
			if not parameter.requires_grad:
				continue
			param = parameter.numel()
			table.add_row([name, param])
			total_params += param 

		print (table)
		print (f'Total trainable parameters: {total_params}')
		return total_params


	def quiver_gradients(self, index, input_tensor, output_tensor, minibatch_size=32):
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
			model = self.model
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

		self.model.train()
		output = self.model(input_tensor)
		output_tensor = output_tensor.reshape(minibatch_size, 1)

		loss = self.loss_function(output, output_tensor)
		self.optimizer.zero_grad() # prevents gradients from adding between minibatches
		loss.backward()

		nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
		self.optimizer.step()

		return output, loss.item()

	def train_model(self):
		"""
		Train the transformer encoder-based model

		Args:
			model: MultiLayerPerceptron object
			optimizer: torch.optim object

		kwargs:
			minibatch_size: int

		Returns:
			None

		"""

		self.model.train()
		count = 0

		for epoch in range(self.epochs):
			pairs = [[i, j] for i, j in zip(self.train_inputs, self.train_outputs)]
			random.shuffle(pairs)
			input_tensors = [i[0] for i in pairs]
			output_tensors = [i[1] for i in pairs]
			total_loss = 0

			for i in range(0, len(pairs) - self.minibatch_size, self.minibatch_size):
				# stack tensors to make shape (minibatch_size, input_size)
				input_batch = torch.stack(input_tensors[i:i + self.minibatch_size])
				output_batch = torch.stack(output_tensors[i:i + self.minibatch_size])

				# skip the last batch if too small
				if len(input_batch) < self.minibatch_size:
					break

				# tensor shape: batch_size x sequence_len x embedding_size
				output, loss = self.train_minibatch(input_batch, output_batch, self.minibatch_size)
				total_loss += loss
				count += 1
				if count % 25 == 0: # plot every 23 epochs for minibatch size of 32
					self.plot_predictions(self.test_inputs, self.test_outputs, count//25)
					# quiver_gradients(count//25, model, input_batch, output_batch)

			print (f'Epoch {epoch} complete: {total_loss} loss')
			
		return

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


	def save_model(self, model):
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


	def evaluate_network(self, model, validation_inputs, validation_outputs):
		"""
		Evaluate network on validation data.

		Args:
			model: Transformer class object
			validation_inputs: arr[torch.Tensor]
			validation_outputs: arr[torch.Tensor]

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
			for i in range(len(validation_inputs)): 
				model_output = model(validation_inputs[i])
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
net = ActivateNetwork()
net.train_model()
net.evaluate_network()




