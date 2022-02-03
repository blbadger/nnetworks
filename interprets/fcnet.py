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

from linear_models import linear_regression

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
		self.softmax = nn.Softmax(dim=1)
		self.layernorm = nn.LayerNorm()


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
		length = len(df['etime'])

		if training:
			df = shuffle(df)
			df.reset_index(inplace=True)

			# 80/20 training/test split
			split_i = int(length * 0.8)

			training = df[:][:split_i]
			self.training_inputs = training[['store_id', 
									'market_id', 
									'created_at',
									'total_busy_dashers', 
									'total_onshift_dashers', 
									'total_outstanding_orders',
									'estimated_store_to_consumer_driving_duration']]
			self.training_outputs = training[['etime']]

			validation_size = length - split_i
			validation = df[:][split_i:split_i + validation_size]
			self.val_inputs = validation[['store_id', 
									'market_id', 
									'created_at',
									'total_busy_dashers', 
									'total_onshift_dashers', 
									'total_outstanding_orders',
									'estimated_store_to_consumer_driving_duration']]
			self.val_outputs = validation[['etime']]

		else:
			self.training_inputs = self.df # not actually training, but matches name for stringify
			self.sequential_count = 0



	def stringify_inputs(self, index):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistant structure to inputs regardless of missing values.

		Args:
			index: int, position of input

		Returns:
			array: string: str of values in the row of interest

		"""


		taken_ls = [4, 1, 4, 3, 3, 3, 4]
		i = index

		string_arr = []
		
		if str(self.training_inputs['created_at'][i]) == '':
			string_arr.append('')
		else:
			string_arr.append(str(self.training_inputs['created_at'][i])[5:7] + str(self.training_inputs['created_at'][i])[8:10])

		fields_ls = ['market_id', 'store_id', 'total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders', 'estimated_store_to_consumer_driving_duration']
		for field in fields_ls:
			if self.training_inputs[field][i] != '':
				if (int(self.training_inputs[field][i])) > 0:
					string_arr.append(str(int(self.training_inputs[field][i])))
				else:
					string_arr.append(str(-int(self.training_inputs[field][i])))
			else:
				string_arr.append('')

		# fill in empty positions with .
		for i in range(len(string_arr)):
			while len(string_arr[i]) < taken_ls[i]:
				string_arr[i] = '.' + string_arr[i]

		string = ''.join(string_arr) + '...'

		return string


	@classmethod
	def string_to_tensor(self, string):
		"""
		Convert a string into a tensor

		Args:
			string: str, input as a string

		Returns:
			tensor
		"""

		places_dict = {s:int(s) for s in '0123456789'}
		places_dict['.'] = 10

		# vocab_size x batch_size x embedding dimension (ie input length)
		tensor_shape = (len(string), 1, 11) 
		tensor = torch.zeros(tensor_shape)

		for i, letter in enumerate(string):
			tensor[i][0][places_dict[letter]] = 1.

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


	def sequential_sample(self):
		"""
		Choose one example from the test set, such that repeated
		choices iterate over the entire set.

		"""

		index = self.sequential_count

		input_string = self.stringify_input(index)
		input_tensor = self.string_to_tensor(input)
		self.sequential_count += 1

		return input, input_tensor


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


def init_transformer(n_letters):
	"""
	Initialize a transformer model

	Args:
		n_letters: int, number of ascii inputs
		emsize: int, the embedding dimension

	Returns:
		model: Transformer object

	"""

	# note that nhead (number of multi-head attention units) must be able to divide feedforward_size (hidden layer size)
	feedforward_size = 280
	nlayers = 3
	nhead = 5
	d_model = 25
	n_output = 1

	model = Transformer(n_output, n_letters, d_model, nhead, feedforward_size, nlayers)
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


def train_model(model, optimizer, minibatch_size=32):
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

		for i in range(0, len(train_input) - minibatch_size, minibatch_size):
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


model = MultiLayerPerceptron(input_size, output_size)
loss_function = nn.CrossEntropyLoss() # integer labels expected
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train_network(model, optimizer, file, minibatch_size=32):
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
	input_samples = [sequential_sample() for i in range(len(inputs))]

	start = time.time()
	for i in range(n_epochs):
		input_samples = input_samples.shuffle()
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


train_network(model, optimizer, file)


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


class Interpret:

	def __init__(self, model, input_tensors, output_tensors, fields):
		self.model = model 
		self.field_array = fields
		self.output_tensors = output_tensors
		self.input_tensors = input_tensors


	def occlusion(self, input_tensor, field_array):
		"""
		Generates a perturbation-type attribution using occlusion.

		Args:
			input_tensor: torch.Tensor
			field_array: arr[int], indicies that mark ends of each field 

		Returns:
			occlusion_arr: array[float] of scores per input index

		"""

		occl_size = 1

		output = self.model(input_tensor)
		zeros_tensor = torch.zeros(input_tensor)
		occlusion_arr = [0 for i in range(len(input_tensor))]
		indicies_arr = []
		total_index = 0

		for i in range(len(field_array)):

			# set all elements of a particular field to 0
			input_copy = torch.clone(input_tensor)
			for j in range(total_index, total_index + field_array[i]):
				input_copy[j] = 0.

			total_index += field_array[i]

			output_missing = self.model(input_copy)

			# assumes a 1-dimensional output
			occlusion = abs(float(output) - float(output_missing))
			indicies_arr.append(i)

		# max-normalize occlusions
		if max(occlusion_arr) != 0:
			correction_factor = 1 / (max(occlusion_arr))
			occlusion_arr = [i*correction_factor for i in occlusion_arr]

		return indicies_arr, occlusion_arr


	def gradientxinput(self, input_tensor, output_shape, model):
		"""
		 Compute a gradientxinput attribution score

		 Args:
		 	input: torch.Tensor() object of input
		 	model: Transformer() class object, trained neural network

		 Returns:
		 	gradientxinput: arr[float] of input attributions

		"""

		# change output to float
		input.requires_grad = True
		output = model.forward(input_tensor)

		# only scalars may be assigned a gradient
		output = output.reshape(1, output_shape).sum()

		# backpropegate output gradient to input
		output.backward(retain_graph=True)

		# compute gradient x input
		final = torch.abs(input_tensor.grad) * input_tensor

		# separate out individual characters
		saliency_arr = []
		s = 0
		for i in range(len(final)):
			if i % 67 ==0 and i > 0: # assumes ASCII character set
				saliency_arr.append(s)
				s = 0
			s += float(final[i])

		# append final element
		saliency_arr.append(s)

		# max norm
		for i in range(len(inputxgrad)):
			maximum = max(inputxgrad[i], maximum)

		# prevent a divide by zero error
		if maximum != 0:
			for i in range(len(inputxgrad)):
				inputxgrad[i] /= maximum

		return inputxgrad


	def heatmap(self, n_observed=100, method='combined'):
		"""
		Generates a heatmap of attribution scores per input element for
		n_observed inputs

		Args:
			n_observed: int, number of inputs
			method: str, one of 'combined', 'gradientxinput', 'occlusion'

		Returns:
			None (saves matplotlib.pyplot figure)

		"""

		if method == 'combined':
			occlusion = self.occlusion(input_tensor)
			gradxinput = self.gradientxinput(input_tensor)
			attribution = [(i+j)/2 for i, j in zip(occlusion, gradxinput)]

		elif method == 'gradientxinput':
			attribution = self.gradientxinput(input_tensor)

		else:
			attribution = self.occlusion(input_tensor)















