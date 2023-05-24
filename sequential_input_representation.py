import math
import os
import time
import pathlib
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn

# import third party libraries
import numpy as np
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from transformers import GPT2Config, GPT2LMHeadModel

# configuration = GPT2Config()
# model = GPT2LMHeadModel(configuration)
# tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2')

load_8bit = False

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# print ('tokenizer downloaded or loaded from cache')

# model = AutoModelForCausalLM.from_pretrained("gpt2", load_in_8bit=load_8bit, device_map='auto')
# print ('model downloaded or loaded from cache')

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def octave(single_input, target_output, iterations, learning_rates, index):
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
	original_input = single_input.clone()
	losses, i_arr = [], []

	for i in range(iterations):
		input_grad, loss = layer_gradient(single_input, target_output, index)
		single_input = single_input.detach()
		single_input -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)*input_grad

	return single_input
 

def generate_expanding_input(model, target, index, lr=0.5): # 0.0007
	"""
	Generates a natural language input representation sequentially
	""" 
	n_tokens = len(target[0])
	full_input = []
	embedding_dim = target.shape[-1]
	starting_input = torch.randn(1, 1, embedding_dim).to(device)
	combined_input = torch.clone(starting_input)
	for i in range(n_tokens):
		combined_input = combined_input.detach().clone()
		single_input = octave(combined_input, target[:, :i+1, :], 500, [lr, lr/100], i)
		random_addition = torch.randn(starting_input.shape).to(device)
		combined_input = torch.cat((combined_input, random_addition), dim=1)

	return combined_input[:, :-1, :]

def generate_sequential_input(model, target, index, lr=0.1):
	"""
	Generates a natural language input representation sequentially
	""" 
	n_tokens = len(target[0])
	full_input = []
	embedding_dim = target.shape[-1]
	starting_input = torch.randn(1, 1, embedding_dim).to(device)
	combined_input = torch.clone(starting_input)
	for i in range(n_tokens):
		random_input = torch.randn(starting_input.shape).to(device)
		focused_target = target[:, i, :]
		print (focused_target.shape)
		single_input = octave(random_input, focused_target, 2000, [lr, lr/10], i)
		combined_input = torch.cat((combined_input, single_input), dim=1)

	return combined_input[:, 1:, :]

def layer_gradient(input_tensor, target, index, cosine_metric=False):
	"""
	Compute the gradient of some layer (chosen previously to be the model output)
	w.r.t the input.

	Args:
		input_tensor: torch.tensor object corresponding to the input
		target: torch.tensor object of the output given the input
		index: int, chooses the layer number to investigate

	Returns:
		gradient: torch.tensor.grad on the input tensor after backpropegation
		loss.item(): float value corresponding to the L1 distance between target and output
	"""
	global load_8bit
	if load_8bit:
		input_tensor = input_tensor.half()

	input_tensor.requires_grad = True
	output = a_model(input_tensor)
	output, target = output.flatten(), target.flatten()
	if cosine_metric:
		loss = torch.abs(torch.dot(output, target)) / (torch.norm(output, p=2) * torch.norm(target, p=2))

	loss = torch.sum(torch.abs(target - output))
	loss.backward()
	gradient = input_tensor.grad

	return gradient, loss.item()


class FCNet(nn.Module):

	def __init__(self, input_length=6, hidden_dim=768):
		super().__init__()
		self.input = nn.Linear(input_length * hidden_dim, 4 * input_length * hidden_dim)
		self.h2h = nn.Linear(4 * input_length * hidden_dim, input_length * hidden_dim)
		self.h2out = nn.Linear(10000, 10000)
		self.gelu = nn.GELU()
		self.input_length = input_length
	
	def forward(self, x):
		x = x.flatten()
		x = self.input(x)
		x = self.gelu(x)

		x = self.h2h(x)
		# x = self.gelu(x)

		# x = self.h2out(x)
		# x = model.lm_head(x.reshape(self.input_length, 768))
		return x

class AbbreviatedGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x: torch.Tensor):
		# Reshape and permute the input tensor

		for i in range(1):

			x = self.model.transformer.h[i](x)[0]

		# x = self.model.lm_head(x)
		return x

class MLPGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x: torch.tensor):

		for i in range(1):
			x = self.model.transformer.h[i].mlp(x)

		return x

class InputGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x):
		embedding = self.model.transformer.wte(x)
  
		for i in range(1):
			x = self.model.transformer.h[i](x)[0]

		return x


prompt = 'The sky is blue.'
tokens = tokenizer.encode(
	  prompt,
	  add_special_tokens=False,
	  return_tensors='pt',
	  max_length = 512,
	  truncation=False,
	  padding=False
	  ).to(device)


model = model.to(device)
embedding = model.transformer.wte(tokens) 
position_ids = torch.tensor([i for i in range(len(tokens))]).to(device)
positional_embedding = model.transformer.wpe(position_ids)
embedding += positional_embedding

shifted_embedding = embedding + 0.05*torch.randn(embedding.shape).to(device)
print (f'Shifted embedding distance: {torch.sum(torch.abs(embedding - shifted_embedding))}')
embedding_weight = model.transformer.wte.weight.float() # convert to float in case model is in 16-bit precision
inverse_embedding = torch.linalg.pinv(embedding_weight)
logits = torch.matmul(shifted_embedding - positional_embedding, inverse_embedding) # invert embedding transformations
tokens = torch.argmax(logits, dim=2)[0]
output = tokenizer.decode(tokens)

a_model = AbbreviatedGPT(model).to(device)
# a_model = MLPGPT(model).to(device)
# a_model = FCNet().to(device)  

a_model.eval()
with torch.no_grad():
	if load_8bit:
		shifted_embedding = shifted_embedding.half()
	shifted_target_tensor = a_model(shifted_embedding).to(device)
	target_tensor = a_model(embedding).to(device)
print (f'Shifted output distance: {torch.sum(torch.abs(shifted_target_tensor - target_tensor))}')

embedding = embedding.detach()
if load_8bit:
	target_tensor = target_tensor.half()

generated_input = generate_sequential_input(a_model, target_tensor, 0)

if load_8bit:
	g_input = generated_input.half()
else:
	g_input = generated_input

generated_target_tensor = a_model(g_input).to(device)
print (f'Generated output distance: {torch.sum(torch.abs(generated_target_tensor - target_tensor))}')
logits = torch.matmul(generated_input - positional_embedding, inverse_embedding)
# tokens = torch.argmax(logits, dim=2)[0]
tokens = torch.topk(logits, 5)[1][0] # indicies of topk of tensor
# print (tokens)

for i in range(5):
	output = tokenizer.decode([o[i] for o in tokens])
	print (output)
print ('\n')

def masked_decode(logits: torch.tensor, allowed_tokens: torch.tensor) -> str:
	"""
	Decode an output tensor via 
	"""
	
	mask_tensor = torch.zeros(logits.shape).to(device)
	# print (mask_tensor.shape)
	mask_tensor[:, :, allowed_tokens] = 1.
	masked_logits = logits * mask_tensor
	allowed_output = torch.argmax(masked_logits, dim=2)[0]
	output = tokenizer.decode(allowed_output)
	print ('Limited output: \n', output)

	return output

allowed_words = 'The sky is blue or red depending on the time of day.'
allowed_tokens = tokenizer.encode(allowed_words)
masked_decode(logits, allowed_tokens)

def check_equality(input, generated_input):
	encoded_input = tokenizer.encode(input, return_tensors='pt').to(device)
	encoded_gen = tokenizer.encode(generated_input, return_tensors='pt').to(device)

	next_token = torch.argmax(model(encoded_input)[0][:, -1, :])
	next_word = tokenizer.decode(next_token)
 
	print (f'Next input word: {next_word}')

	next_gen_token = torch.argmax(model(encoded_gen)[0][:, -1, :])
	next_gen_word = tokenizer.decode(next_gen_token)
	print (f'Next generated input word: {next_gen_word}')
 
	return next_word, next_gen_word

# check_equality(prompt, output)
