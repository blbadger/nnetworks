# fcnet.py


import numpy as np 
import pandas as pd 
import random
import torch 
import torch.nn as nn

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
		softout = self.softmax(out)

		return softout 

model = MultiLayerPerceptron(input_size, output_size)
loss_function = nn.CrossEntropyLoss() # integer labels expected
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


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


def test_model(test_inputs, test_outputs, model):
	"""
	Test the model on the validation set

	Args:
		test_inputs: torch.Tensor
		test_outputs: torch.Tensor of expected values
		model: MultiLayerPerceptron object instance

	Returns:
		None

	"""
	model.eval()
	correct_count = 0

	for i in range(test_inputs):
		inp = test_inputs[i]
		output = model(inp)

		true_out = test_outputs[i]
		if output == true_out:
			correct_count += 1

	print (f'Test complete: {correct_count / len(test_inputs)} accuracy')
	return

