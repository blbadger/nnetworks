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
from prettytable import PrettyTable
from google.colab import files
from google.colab import drive
drive.mount('/content/gdrive')

# files.upload() # upload flower_photos_2
# !unzip flower_photos_2.zip

# dataset directory specification
data_dir = pathlib.Path('flower_photos_2',  fname='Combined')

image_count = len(list(data_dir.glob('*/*.jpg')))
class_names = [item.name for item in data_dir.glob('*') 
			   if item.name not in ['._.DS_Store', '._DS_Store', '.DS_Store']]

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class ImageDataset(Dataset):
	"""
	Creates a dataset from images classified by folder name.  Random
	sampling of images to prevent overfitting
	"""

	def __init__(self, img_dir, transform=None, target_transform=None, image_type='.png'):
		# specify image labels by folder name 
		self.img_labels = [item.name for item in data_dir.glob('*')]

		# construct image name list: randomly sample images for each epoch
		images = list(img_dir.glob('*/*' + image_type))
		random.shuffle(images)
		self.image_name_ls = images[:800]

		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.image_name_ls)

	def __getitem__(self, index):
		# path to image
		img_path = os.path.join(self.image_name_ls[index])
		image = torchvision.io.read_image(img_path) # convert image to tensor of ints , torchvision.io.ImageReadMode.GRAY
		image = image / 255. # convert ints to floats in range [0, 1]
		image = torchvision.transforms.Resize(size=[299, 299])(image)
		# image = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)	

		# assign label to be a tensor based on the parent folder name
		label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

		# convert image label to tensor
		label_tens = torch.tensor(self.img_labels.index(label))
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image, label_tens

def clip_gradient(input_grad):
	"""
	Perform maximum and minimum clipping on the input gradient

	Args:
		input_grad: torch.tensor object of the gradient to be clipped

	Returns;
		input_grad: torch.tensor object
	"""

	mean = torch.mean(input_grad)
	stdev = torch.std(input_grad)
	input_grad = torch.clip(input_grad, mean - stdev, mean + stdev)
	return input_grad

def save_image(single_input, count):
	"""
	Saves a .png image of the single_input tensor

	Args:
		single_input: torch.tensor of the input 
		count: int, class number

	Returns:
		None (writes .png to storage)
	"""

	print (count)
	plt.figure(figsize=(10, 10))
	image_width = len(single_input[0][0])
	target_input = single_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
	plt.axis('off')
	plt.imshow(target_input)
	images_dir = '/content/gdrive/My Drive/googlenet_generation_nopad'
	plt.savefig("{}".format(images_dir) + "/Class {0:04d}- ".format(count) + "{}.png".format(imagenet_classes[count]), bbox_inches='tight', pad_inches=0, dpi=390)
	plt.close()
	return

def random_crop(input_image, size):
	"""
	Crop an image with a starting x, y coord from a uniform distribution

	Args:
		input_image: torch.tensor object to be cropped
		size: int, size of the desired image (size = length = width)

	Returns:
		input_image_cropped: torch.tensor
		crop_height: starting y coordinate
		crop_width: starting x coordinate
	"""

	image_width = len(input_image[0][0])
	image_height = len(input_image[0])
	crop_width = random.randint(0, image_width - size)
	crop_height = random.randint(0, image_width - size)
	input_image_cropped = input_image[:, :, crop_height:crop_height + size, crop_width: crop_width + size]

	return input_image_cropped, crop_height, crop_width


def octave(single_input, target_output, iterations, learning_rates, sigmas, size, pad=False, crop=True):
	"""
	Perform an octave (scaled) gradient descent on the input.

	Args;
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
		if crop:
			cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
		else:
			cropped_input, crop_height, crop_width = random_crop(single_input.detach(), len(single_input[0][0]))
			size = len(single_input[0][0])
		single_input = single_input.detach() # remove the gradient for the input (if present)
		input_grad = layer_gradient(Inception, cropped_input, target_output) # compute input gradient
		single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)* input_grad # gradient descent step
		single_input = torchvision.transforms.functional.gaussian_blur(single_input, 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations))
		if pad:
			single_input = torchvision.transforms.Pad([1, 1], fill=0.7)(single_input)

	return single_input


def generate_singleinput(model, input_tensors, output_tensors, index, count, random_input=True):
	"""
	Generates an input for a given output

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int, target class index to generate
		cout: int, time step

	kwargs: 
		random_input: bool, if True then a scaled random normal distributionis used

	returns:
		None (saves .png image)
	"""

	manualSeed = 999
	random.seed(manualSeed)
	torch.manual_seed(manualSeed)

	class_index = index

	if random_input:
		single_input = (torch.randn(1, 3, 299, 299))/20 + 0.7 # scaled normal distribution initialization

	else:
		single_input = input_tensors[0]
 
	single_input = single_input.to(device)
	original_input = torch.clone(single_input).reshape(3, 299, 299).permute(1, 2, 0).cpu().detach().numpy()
	single_input = single_input.reshape(1, 3, 299, 299)
	original_input = torch.clone(single_input).reshape(3, 299, 299).permute(1, 2, 0).cpu().detach().numpy()
	target_output = torch.tensor([class_index], dtype=int)
 
	pad = False
	if pad:
		single_input = octave(single_input, target_output, 220, [6, 5], [2.4, 0.8], 0, pad=True, crop=False)
	else:
		single_input = octave(single_input, target_output, 220, [1.5, 1], [2.4, 0.8], 0, pad=False, crop=False)

	single_input = torchvision.transforms.Resize([490, 490])(single_input)
	# single_input = octave(Inception, single_input, target_output, 1500, [1.7, 0.3], [1.5, 0.3], 475, pad=False, crop=True) 
	single_input = octave(single_input, target_output, 500, [1.3, 0.45], [1.5, 0.4], 470, pad=False, crop=True) # 450 image size for pad

	save_image(single_input, index)
	return single_input


def layer_gradient(model, input_tensor, desired_output):
	"""
	Compute the gradient of the output (logits) with respect to the input 
	using an L1 metric to maximize the target classification.

	Args:
		model: torch.nn.model
		input_tensor: torch.tensor object corresponding to the input image
		true_output: torch.tensor object of the desired classification label

	Returns:
		gradient: torch.tensor.grad on the input tensor after backpropegation

	"""
	input_tensor.requires_grad = True
	output = model(input_tensor)
	loss = torch.abs(200 - output[0][int(desired_output)]) # no L1 regularization needed
	loss.backward()
	gradient = input_tensor.grad

	return gradient


def generate_inputs(model, count=0):
	"""
	Generate the output of each desired class

	Args:
		model: torch.nn.Module object of interest

	kwargs:
		count: int, time step 

	Returns:
		None (saves .png images to storage)
	"""

	for i in range(1000):
		generate_singleinput(model, [], [], i, count, random_input=True)

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
			plt.imshow(input_batch[n])
		plt.tight_layout()

	plt.tight_layout()
	plt.savefig('transformed_flowers{0:04d}.png'.format(count), dpi=410)
	plt.close()
	return

def count_parameters(model):
	"""
	Display the tunable parameters in the model of interest

	Args:
		model: torch.nn object

	Returns:
		total_params: the number of model parameters

	"""

	table = PrettyTable(['Module', 'Parameters'])
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		param = parameter.numel()
		table.add_row([name, param])
		total_params += param 

	print (table)
	print (f'Total trainable parameters: {total_params}')
	return total_params


loss_fn = nn.CrossEntropyLoss()

Inception = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True, init_weights=True).to(device)
Inception.eval()
generate_inputs(Inception, 0)

