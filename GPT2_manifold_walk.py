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
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2')  
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2')

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

# manualSeed = 999
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)


def layer_gradient(model, input_tensor, flat=True):
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

	input_tensor.requires_grad = True
	output = a_model(input_tensor)
	output.backward(gradient=torch.ones(output.shape).to(device))
	gradient = torch.clone(input_tensor.grad)

	# input_tensor.grad = None
	# output = a_model(input_tensor)
	# sout = torch.sum(output)
	# sout.backward()
	# gradient2 = torch.clone(input_tensor.grad)

	# print (gradient2 - gradient)
	# input_tensor.grad = None
	# gradient3 = torch.zeros(input_tensor.shape).to(device)
	# output = a_model(input_tensor)
	# if flat:
	# 	for i in range(len(output)):
	# 		output = a_model(input_tensor) 
	# 		loss = output[i]
	# 		loss.backward()
	# 		gradient3 += input_tensor.grad
	# 		print (i)
	# 		input_tensor.grad = None


	# else:
	# 	for i in range(len(output[0])):
	# 		for j in range(len(output[0][0])):
	# 				output = a_model(input_tensor)
	# 				loss = output[0, i, j]
	# 				loss.backward()
	# 				gradient2 += input_tensor.grad

	# print (gradient - gradient2)
	return gradient, output

def target_gradient(model, input_tensor, target_output):
	"""

	"""
	input_tensor.requires_grad = True
	output = a_model(input_tensor)
	loss = torch.sum(torch.abs(output - target_output))
	print (loss.item())
	loss.backward()
	gradient = input_tensor.grad
	return gradient

class FCNet(nn.Module):

	def __init__(self, input_length=5, hidden_dim=768):
		super().__init__()
		self.input = nn.Linear(input_length*hidden_dim, 4*input_length*hidden_dim)
		self.h2h = nn.Linear(4*input_length*hidden_dim, input_length*hidden_dim)
		self.gelu = nn.GELU()
		self.input_length = input_length
	
	def forward(self, x):
		x = x.flatten()
		x = self.input(x)
		x = self.gelu(x)

		x = self.h2h(x)
		return x

class AbbreviatedGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model
	
	def forward(self, x: torch.Tensor):
		# Reshape and permute the input tensor

		for i in range(12):
			x = self.model.transformer.h[i](x)[0]

		# x = self.model.lm_head(x)
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


def choose_model(choice):
	if choice == 'GPT':
		a_model = AbbreviatedGPT(model).double().to(device)
	else:
		a_model = FCNet().to(device)
	return a_model

a_model = choose_model('GPT')
a_model.eval()

model = model.to(device)
embedding = model.transformer.wte(tokens) 
position_ids = torch.tensor([i for i in range(len(tokens))]).to(device)
positional_embedding = model.transformer.wpe(position_ids)
embedding += positional_embedding

shifted_embedding = embedding + 0.05*torch.randn(embedding.shape).to(device)
print (f'Shifted embedding distance: {torch.sum(torch.abs(embedding - shifted_embedding))}')
embedding_weight = model.transformer.wte.weight
inverse_embedding = torch.linalg.pinv(embedding_weight)
logits = torch.matmul(shifted_embedding - positional_embedding, inverse_embedding) # invert embedding transformations
tokens = torch.argmax(logits, dim=2)[0]
output = tokenizer.decode(tokens)
print (output)


with torch.no_grad():
	shifted_target_tensor = a_model(shifted_embedding).to(device)
	target_tensor = a_model(embedding).to(device)
print (f'Shifted output distance: {torch.sum(torch.abs(shifted_target_tensor - target_tensor))}')

original_embedding = torch.clone(embedding)

def tangent_walk(embedding):
	for i in range(1):
		embedding = embedding.double().detach()
		gradient, _ = layer_gradient(a_model, embedding)
		gradient = torch.squeeze(gradient, dim=0) # remove batch dim
		perp_vector = torch.linalg.svd(gradient).Vh[-1] # any index greater than input_length
		original_embedding = embedding.clone()
		embedding = embedding + 0.1*perp_vector
		print (gradient @ perp_vector) # check for orthogonality via mat mult

	return embedding

def clamped_walk(embedding):
	with torch.no_grad():
		target_output = a_model(embedding)
	embedding = embedding.detach()

	for i in range(10):
		clamped = torch.randint(768, (1,))
		shape = embedding[:, :, clamped:].shape
		embedding[:, :, clamped:] += 0.0002*torch.rand(shape).to(device)
		for i in range(31):
			gradient = target_gradient(a_model, embedding, target_output)
			embedding = embedding.detach()
			embedding[:, :, :clamped] = embedding[:, :, :clamped] - 0.00002*gradient[:, :, :clamped]
		
	return embedding


gen_embedding = tangent_walk(embedding)
generated_target_tensor = a_model(gen_embedding).to(device)
print (f'Generated Input Distance: {torch.sum(torch.abs(gen_embedding - original_embedding))}')
print (f'Generated Output distance: {torch.sum(torch.abs(generated_target_tensor - target_tensor))}')
logits = torch.matmul(gen_embedding - positional_embedding, inverse_embedding)
# tokens = torch.argmax(logits, dim=2)[0]
tokens = torch.topk(logits, 5)[1][0] # indicies of topk of tensor
# print (tokens)
# for i in range(5):
# 	output = tokenizer.decode([o[i] for o in tokens])
# 	print (output)

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
