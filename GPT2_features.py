# import standard libraries
import time
import pathlib
import os
import random

# import third party libraries
import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

def octave(single_input, target_output, iterations, learning_rates, sigmas, size, index, crop=True):
	"""
	Perform an octave (scaled) gradient descent on the input.

	Args:
		single_input: torch.tensor of the input
		target_output: torch.tensor of the desired output category
		iterations: int, the number of iterations desired
		learning_rates: arr[int, int], pair of integers corresponding to start and end learning rates
		sigmas: arr[int, int], pair of integers corresponding to the start and end Gaussian blur sigmas
		size: int, desired dimension of output image (size = length = width)

	kwargs:
		pad: bool, if True then padding is applied at each iteration of the octave
		crop: bool, if True then gradient descent is applied to cropped sections of the input

	Returns:
		single_input: torch.tensor of the transformed input
	"""

	start_lr, end_lr = learning_rates
	start_sigma, end_sigma = sigmas

	for i in range(iterations):
		# single_input = single_input.detach() # remove the gradient for the input (if present)
		input_grad = layer_gradient(model, single_input, target_output, index) # compute input gradient
		single_input = single_input.detach()
		single_input -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)* input_grad # gradient descent step

	return single_input

def generate_singleinput(model, input, index):
	"""
	Generates an input for a given output

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int, index of input desired
		count: int, time step
	kwargs: 
		random_input: bool, if True then a random input is used (index is ignored)

	returns:
		None (saves .png image)
	"""
	manualSeed = 999
	random.seed(manualSeed)
	torch.manual_seed(manualSeed)
	single_input = input
	single_input = octave(single_input, [], 100, [0.5, 0.4], [2.4, 0.8], 0, index)
    
	return single_input

def layer_gradient(model, input_tensor, desired_output, index):
	"""
	Compute the gradient of some layer (chosen previously to be the model output)
	w.r.t the input.

	Args:
		model: torch.nn.model
		input_tensor: torch.tensor object corresponding to the input image
		desired_output: torch.tensor object of the desired classification label
		index: int, chooses the layer number to investigate

	Returns:
		gradient: torch.tensor.grad on the input tensor after backpropegation
	"""

	input_tensor.requires_grad = True
	output = a_model(input_tensor).to(device)
	focus = output[:, :, index]
	target = torch.ones(focus.shape).to(device)*200
	loss = 0.1*torch.sum(target - focus)
	loss.backward()
	gradient = input_tensor.grad

	return gradient

class AbbreviatedGPT(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor

        for i in range(1):
            x = self.model.transformer.h[i](x)[0]

        return x

class InputGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model
	
	def forward(self, x):
		embedding = self.model.transformer.wte(x)
		for i in range(4):
			x = self.model.transformer.h[i](x)[0]


class OutputGPT(nn.Module):

	def __init__(self, model, output_head):
		super().__init__()
		self.model = model
		self.output_head = output_head
	
	def forward(self, x):
		# for i in range(1, 12):
		# 	x = self.model.transformer.h[i](x)[0]

		x = self.model.lm_head(x)
		return self.output_head(loss=None, logits=x)

prompt = 'This is the initial prompt: A very bland sentence that is not descriptive.'
tokens = tokenizer.encode(
	  prompt,
	  add_special_tokens=False,
	  return_tensors='pt',
	  max_length = 512,
	  truncation=False,
	  padding=False
	  )
print (tokens)
# embeddings = model.transformer.get_input_embeddings(tokens)
embedding = model.transformer.wte(tokens)
embedding_weight = model.transformer.wte.weight
inverse_embedding = torch.linalg.pinv(embedding_weight)
print (inverse_embedding.shape, embedding.shape)
logits = torch.matmul(embedding, inverse_embedding)
tokens = torch.argmax(logits, dim=2)[0]
output = tokenizer.decode(tokens)
print (output)
 
a_model = AbbreviatedGPT(model)
a_model.eval()
for i in range(16):
	random_input = torch.randn((1, 20, 768))
	embedding = embedding.detach()
	generated_input = generate_singleinput(a_model, embedding, i)
	logits = torch.matmul(generated_input, inverse_embedding)
	tokens = torch.argmax(logits, dim=2)[0]
	output = tokenizer.decode(tokens)
	print (output)
	# causal_lm_output = CausalLMOutputWithCrossAttentions
	# outGPT = OutputGPT(model, causal_lm_output)

	# output = outGPT(generated_input).logits
	# output = torch.argmax(output, dim=2)[0]
	# output = tokenizer.decode(output)
	# output = tokenizer.decode(generated_input)
	# print (output)
