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
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

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
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from transformers import GPT2Config, GPT2LMHeadModel

# configuration = GPT2Config()
# model = GPT2LMHeadModel(configuration)
# tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2-xl')


load_8bit = False

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print ('tokenizer downloaded or loaded from cache')


model = AutoModelForCausalLM.from_pretrained("gpt2-xl", load_in_8bit=load_8bit, device_map='auto')
print ('model downloaded or loaded from cache')

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
		input_grad, loss = layer_gradient(model, single_input, target_output, index)
		single_input = single_input.detach()
		single_input -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)*input_grad

	return single_input


def layer_gradient(model, input_tensor, target, index):
	"""
	Compute the gradient of some layer (chosen previously to be the model output)
	w.r.t the input.

	Args:
		model: torch.nn.model
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


class InputGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x: torch.tensor) -> torch.tensor:
		# embedding = self.model.transformer.wte(x)

		# replaces wte transformation
		x = torch.matmul(x, self.model.transformer.wte.weight)

		o1 = self.model.transformer.h[0](x)[0]
		o2 = self.model.transformer.h[0](o1)[0]
		o3 = self.model.transformer.h[0](o2)[0]
  
		# for i in range(3):
		# 	x = self.model.transformer.h[i](x)[0]

		return o1, o2, o3


prompt = 'This is a prompt sentence'
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
target_logits = torch.matmul(embedding - positional_embedding, inverse_embedding) 

target_logits = torch.zeros(logits.shape).to(device)
for i in range(len(tokens[0])):
	target_logits[:, i, tokens[0, i]] = 10
# print (torch.max(target_logits), torch.min(torch.abs(target_logits), target_logits))

tokens = torch.argmax(target_logits, dim=2)[0]
output = tokenizer.decode(tokens)
print (output)

a_model = InputGPT(model).to(device)
a_model.eval()

shifted_input = target_logits + 0.05*torch.randn(target_logits.shape).to(device)
with torch.no_grad():
	shifted_output = a_model(shifted_input)
	output = a_model(target_logits)
print (f'Shifted Output Distance: {torch.sum(torch.abs(shifted_output - output))}')


def generate_logits(model, target_logits, target_output, lr=0.01):
	random_input = torch.randn(target_logits.shape).to(device)
	single_input = octave(random_input, target_output, 500, [lr, lr/10], 0)
	return single_input

embedding = embedding.detach()
if load_8bit:
	target_tensor = target_tensor.half()

with torch.no_grad():
	target_tensor = a_model(target_logits)

# generated_input = generate_singleinput(a_model, target_tensor, 0)
generated_input = generate_logits(a_model, target_logits, target_tensor)

if load_8bit:
	g_input = generated_input.half()
else:
	g_input = generated_input

generated_target_tensor = a_model(g_input).to(device)
print (f'Generated output distance: {torch.sum(torch.abs(generated_target_tensor - target_tensor))}')
# logits = torch.matmul(generated_input - positional_embedding, inverse_embedding)
# tokens = torch.argmax(logits, dim=2)[0]
tokens = torch.topk(generated_input, 5)[1][0] # indicies of topk of tensor

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

allowed_words = 'This is a prompt sentence with some extra words attached to increase the allowed vocabulary by a small margin'
allowed_tokens = tokenizer.encode(allowed_words)
masked_decode(generated_input, allowed_tokens)

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

check_equality(prompt, output)
