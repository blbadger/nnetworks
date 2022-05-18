# inputgen_inception.py
# InceptionV3 applied to image generation using the input gradient descent

# import standard libraries
import time
import pathlib
import os
import pandas as pd 
import random

# import third party libraries
import numpy as np 
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt


# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def generate_input(model, input_tensors, output_tensors, index, count, random_input=True):
	"""
	Generates an input for a given output

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int

	returns:
		None (saves .png image)
	"""
	# manualSeed = 999
	# random.seed(manualSeed)
	# torch.manual_seed(manualSeed)

	class_index = 292

	if random_input:
		single_input = (torch.rand(1, 3, 256, 256))/25 + 0.5 # scaled normal distribution initialization

	else:
		single_input = input_tensors[0]
 
	single_input = single_input.to(device)
	original_input = torch.clone(single_input).reshape(3, 256, 256).permute(1, 2, 0).cpu().detach().numpy()
	single_input = single_input.reshape(1, 3, 256, 256)
	original_input = torch.clone(single_input).reshape(3, 256, 256).permute(1, 2, 0).cpu().detach().numpy()
	target_output = torch.tensor([class_index], dtype=int)

	for i in range(100):
		single_input = single_input.detach() # remove the gradient for the input (if present)
		predicted_output = model(single_input)
		# print (predicted_output.argmax())
		input_grad = layer_gradient(model, single_input, target_output) # compute input gradient
		# input_grad /= (torch.std(input_grad) + 1e-8)
		# input_grad = torch.clip(input_grad,-1, 1)
		
		single_input = single_input - 0.15 * input_grad # gradient descent step
		# single_input = single_input - (0.1/(i+1)) * input_grad # gradient descent step
		# single_input = convolution(single_input)
		if i < 76:
			single_input = torchvision.transforms.functional.gaussian_blur(single_input, 3)
			if i % 5 == 0:
				single_input = torch.nn.functional.interpolate(single_input, 256)
			elif i % 5 == 1:
				single_input = torch.nn.functional.interpolate(single_input, 128)
			elif i % 5 == 2:
				single_input = torch.nn.functional.interpolate(single_input, 160)
			elif i % 5 == 3:
				single_input = torch.nn.functional.interpolate(single_input, 100)
			elif i % 5 == 4:
				single_input = torch.nn.functional.interpolate(single_input, 200)

		single_input = torchvision.transforms.ColorJitter(0.0001)(single_input)
		# if i % 10 < 5:
		# 	single_input = torchvision.transforms.functional.rotate(single_input, 36)
		# else:
		# 	single_input = torchvision.transforms.functional.rotate(single_input, -36)
		# single_input = torchvision.transforms.RandomAffine(0.0001, [0.0001, 0.0001])(single_input)


	single_input = single_input.clone() / torch.max(single_input)
	target_input = single_input.reshape(3, 256, 256).permute(1, 2, 0).cpu().detach().numpy()
	plt.figure(figsize=(15, 10))
	plt.subplot(1, 2, 1)
	plt.axis('off')
	plt.imshow(original_input, alpha=1)
	plt.tight_layout()

	plt.subplot(1, 2, 2)
	plt.axis('off')
	plt.imshow(target_input, alpha=1)
	plt.tight_layout()
	plt.show()

	# plt.savefig('adversarial_example{0:04d}.png'.format(count), dpi=410)
	plt.close()

	return target_input

def loss_gradient(model, input_tensor, true_output, output_dim):
	"""
	 Computes the gradient of the input wrt. the objective function

	 Args:
		input: torch.Tensor() object of input
		model: Transformer() class object, trained neural network

	 Returns:
		gradientxinput: arr[float] of input attributions

	"""

	# change output to float
	true_output = true_output.reshape(1)
	input_tensor.requires_grad = True
	output = model.forward(input_tensor)
	loss = loss_fn(output, true_output)

	# backpropegate output gradient to input
	loss.backward(retain_graph=True)
	gradient = input_tensor.grad
	return gradient


def layer_gradient(model, input_tensor, true_output):
	"""

	"""
	# model.fc.register_forward_hook(get_activation('fc'))
	input_tensor.requires_grad = True
	output = model(input_tensor)
	# output_activations = activation['fc']

	loss = torch.abs(200 - output[0][int(true_output)]) + 0.001 * torch.abs(input_tensor).sum() # maximize output val and minimize L1 norm of the input
	# print (loss)
	# print (output[0][int(true_output)])
	# print (max([i for i in output[0][:]]))
	loss.backward()
	gradient = input_tensor.grad
	return gradient


def adversarial_test(dataloader, model, count=0):
	model.eval()
	ls = []
	for i in range(16):
		ls.append(generate_input(model, [], [], i, count=count))
	show_batch(ls, grayscale=True)
	model.train()
	return

def show_batch(input_batch, count=0, grayscale=False):
	"""
	Show a batch of images with gradientxinputs superimposed

	Args:
		input_batch: arr[torch.Tensor] of input images
		output_batch: arr[torch.Tensor] of classification labels
		gradxinput_batch: arr[torch.Tensor] of attributions per input image
	kwargs:
		individuals: Bool, if True then plots 1x3 image figs for each batch element
		count: int

	returns:
		None (saves .png img)

	"""

	plt.figure(figsize=(15, 15))
	for n in range(16):
		ax = plt.subplot(4, 4, n+1)
		plt.axis('off')
		if grayscale:
			plt.imshow(input_batch[n], cmap='gray')
		else:
			print (input_batch[n])
			plt.imshow(input_batch[n])
		plt.tight_layout()

	plt.tight_layout()
	# plt.savefig('original_flowers{0:04d}.png'.format(count), dpi=410)
	plt.show()
	plt.close()
	return

# file = wget.download('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt')

# Read the categories
# with open("imagenet_classes.txt", "r") as f:
#     class_names = [s.strip() for s in f.readlines()]


loss_fn = nn.CrossEntropyLoss()
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)

model.eval()
adversarial_test(None, model, 0)


