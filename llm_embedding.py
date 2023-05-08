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
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("gpt2")
print ('tokenizer downloaded or loaded from cache')

load_8bit = True
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


def generate_singleinput(model, target, index, lr=0.005): # 0.0007
	"""
	Generates an input for a given output

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
	kwargs: 
		lr: float, learning rate

	returns:
		None (saves .png image)
	"""
	random_input = torch.randn(embedding.shape).to(device)
	single_input = octave(random_input, target, 100, [lr, lr/10], index)
	
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

class AbbreviatedGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x: torch.Tensor):
		# Reshape and permute the input tensor

		for i in range(24):
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
generated_input = generate_singleinput(a_model, target_tensor, 0)
generated_target_tensor = a_model(generated_input).to(device)
print (f'Generated output distance: {torch.sum(torch.abs(generated_target_tensor - target_tensor))}')
logits = torch.matmul(generated_input - positional_embedding, inverse_embedding)
# tokens = torch.argmax(logits, dim=2)[0]
tokens = torch.topk(logits, 5)[1][0] # indicies of topk of tensor
# print (tokens)


for i in range(5):
	output = tokenizer.decode([o[i] for o in tokens])
	print (output)

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



def kmeans_cluster(embeddings, n_clusters, *labels):
	kmeans = KMeans(n_clusters=n_clusters)
	kmeans.fit(embeddings)
	return kmeans.labels_


def plot_pca(embeddings, components=2):
	pca = PCA(n_components=components, svd_solver='full')
	rows = [i for i in range(len(embeddings))] # assumes squeezed embedding
	pca.fit(embeddings)
	new_output = pca.transform(embeddings)
	x, y = [i[0] for i in new_output], [i[1] for i in new_output]
	plt.scatter(x, y)
	# label inputs
	for i, txt in enumerate(rows):
		plt.annotate(rows[i], (x[i], y[i]))

	plt.savefig('pca_plot.png', dpi=300)
	plt.close()
	return


def tsne_with_kmeans_labels(embeddings, n_clusters=4):
	labels = kmeans_cluster(o)
	tsne = TSNE(n_components=2, init='random', perplexity=30)
	tsne.fit(embeddings)
	new_output = tsne.fit_transform(embeddings)

	plt.figure(figsize=(10, 10), dpi=300)
	for label in range(n_clusters):
		final_output = [new_output[i] for i in range(len(new_output)) if labels[i] == label]
		x, y = [i[0] for i in final_output], [i[1] for i in final_output]
		plt.scatter(x, y, label=label, alpha=0.5)
		for i in range(len(final_output)):
			plt.annotate(label, (x[i], y[i]))

	plt.legend()
	plt.savefig('tsne_figure.png', dpi=300, layout='tight')
	plt.close()
	return


embedding = e_model(input_tokens)
plot_pca(embedding, components=2)
kmeans_cluster(embedding, 2)
tsne_with_kmeans_labels(embedding)