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

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

# manualSeed = 999
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

def layer_gradient(model, input_tensor):
	input_tensor.requires_grad = True
	output = a_model(input_tensor)
	output.backward(gradient=torch.ones_like(output).to(device))
	gradient = input_tensor.grad.detach().clone()
	return gradient

 

def individual_gradient(model, input_tensor):
	with torch.no_grad():
		input_tensor = input_tensor.flatten()
	input_tensor.requires_grad = True
	gradient = torch.zeros(input_tensor.shape).double().to(device)
	output = a_model(input_tensor)
	for i in range(len(output)):
		output = a_model(input_tensor) 
		loss = output[i]
		loss.backward()
		gradient += input_tensor.grad
		input_tensor.grad = None

	return gradient


class FCNet(nn.Module):

	def __init__(self, input_dim=10, hidden_dim=10):
		super().__init__()
		self.input = nn.Linear(input_dim, hidden_dim)
		self.h2o = nn.Linear(hidden_dim, input_dim)
		self.gelu = nn.GELU()
	
	def forward(self, x):
		x = x.flatten()
		x = self.input(x)
		x = self.gelu(x)

		x = self.h2o(x)
		return x


def tangent_walk(embedding):
	for i in range(1):
		embedding = embedding.double().detach()
		gradient = layer_gradient(a_model, embedding)
		embedding.grad = None # zero out grads on embedding tensor
		s_grad = summed_gradient(a_model, embedding)
		embedding.grad = None
		i_grad = individual_gradient(a_model, embedding)
		perp_vector = torch.linalg.svd(gradient).Vh[-1] # any index greater than input_length
		original_embedding = embedding.clone()
		embedding = embedding + 0.1*perp_vector
		print (gradient @ perp_vector) # check for orthogonality via mat mult

	return embedding

size = 1000
a_model = FCNet(input_dim=size, hidden_dim=size).to(device)
a_model = a_model.double()
embedding = torch.randn(1, size).double().to(device)
original_embedding = torch.clone(embedding).double()
target_tensor = torch.clone(a_model(embedding))
gen_embedding = tangent_walk(embedding)
generated_target_tensor = a_model(gen_embedding).to(device)
print (f'Generated Input Distance: {torch.sum(torch.abs(gen_embedding - original_embedding))}')
print (f'Generated Output distance: {torch.sum(torch.abs(generated_target_tensor - target_tensor))}')


def check_grads():
	embedding = torch.randn(size).double().to(device)
	embedding.requires_grad = True
	weights = a_model.input.weight
	output = torch.matmul(weights, embedding) + a_model.input.bias
	y = torch.sum(output)
	y.backward()
	summed_grad = embedding.grad.detach().clone()

	embedding.grad = None
	output = torch.matmul(weights, embedding) + a_model.input.bias
	output.backward(gradient=torch.ones_like(output))
	grad = embedding.grad.detach().clone()
	print (embedding.grad == summed_grad)

	gradient = torch.zeros(embedding.shape).to(device)
	for i in range(len(output)):
		embedding = torch.randn(size).double().to(device)
		embedding.requires_grad = True
		output = torch.matmul(weights, embedding) + a_model.input.bias
		output[i].backward()
		gradient += embedding.grad

	print (gradient - summed_grad < 1e-5)
	return

# check_grads()
