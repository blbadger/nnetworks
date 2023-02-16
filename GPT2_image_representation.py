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

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

# dataset directory specification
data_dir = pathlib.Path('../tesla',  fname='Combined')
# data_dir = pathlib.Path('dalmatians', fname='Combined')

image_count = len(list(data_dir.glob('*/*.jpg')))
print (image_count)

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
		# random.shuffle(images)
		self.image_name_ls = images[:800]

		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.image_name_ls)

	def __getitem__(self, index):
		# path to image
		img_path = os.path.join(self.image_name_ls[index])
		image = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB) # convert image to tensor of ints , torchvision.io.ImageReadMode.GRAY
		image = image / 255. # convert ints to floats in range [0, 1]
		image = torchvision.transforms.Resize(size=[224, 224])(image)

		# assign label to be a tensor based on the parent folder name
		label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

		# convert image label to tensor
		label_tens = torch.tensor(self.img_labels.index(label))
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image, label_tens

# specify batch size
minibatch_size = 1
data = ImageDataset(data_dir, image_type='.jpg')
dataloader = DataLoader(data, batch_size=minibatch_size, shuffle=False)


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

def octave(single_input, target_output, iterations, learning_rates, size, index, crop=True):
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

	for i in range(iterations):
		if crop:
			cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
		else:
			cropped_input, crop_height, crop_width = random_crop(single_input.detach(), len(single_input[0][0]))
			size = len(single_input[0][0])
		single_input = single_input.detach() # remove the gradient for the input (if present)
		input_grad = layer_gradient(new_vision, cropped_input, target_output, index) # compute input gradient
		single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)* input_grad # gradient descent step

	return single_input

def generate_singleinput(model, input_tensors, target_output, index, count):
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
	dim = len(input_tensors[0][0])

	target_input = input_tensors.to(device).reshape(1, 3, 224, 224)
	single_input = (torch.randn(1, 3, dim, dim)/20 + 0.7).to(device)
	single_input = single_input.to(device)
	single_input = octave(single_input, target_output, 1000, [1e-3, 1e-4], 0, index, crop=False)

	output = model(single_input).to(device)
	print (f'L2 distance between target and generated image: {torch.sqrt(torch.sum((target_tensor - output)**2))}')
	target_input = torch.tensor(target_input).reshape(1, 3, dim, dim).to(device)
	input_distance = torch.sqrt(torch.sum((single_input - image)**2))
	print (f'L2 distance on the input: {input_distance}')

	plt.figure(figsize=(10, 10))
	image_width = len(single_input[0][0])
	final_input= single_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy() 
	plt.axis('off')
	plt.imshow(final_input)
	plt.savefig('figure', bbox_inches='tight', pad_inches=0.1, transparent=True)
	plt.close()
	return  

	return target_input

def layer_gradient(model, input_tensor, target_output, index):
	"""
	Compute the gradient of some layer (chosen previously to be the model output)
	w.r.t the input.

	Args:
		model: torch.nn.model
		input_tensor: torch.tensor object corresponding to the input image
		desired_output: torch.tensor object of the desired output layer activation
						to be inverted
		index: int, chooses the layer number to investigate

	Returns:
		gradient: torch.tensor.grad on the input tensor after backpropegation
	"""

	input_tensor.requires_grad = True
	output = model(input_tensor).to(device)
	loss = torch.sum(torch.abs(output - target_output)) # L1 loss on output
	loss.backward()
	gradient = input_tensor.grad

	return gradient

def generate_input(dataloader, model, target_tensor, count=0):
	"""
	Intermediary function to generate a deep dream output

	Args:
		dataloader: torch.utils.DataLoader() object
		model: torch.nn.Module(), model used to generate output

	kwargs:
		count: int, time step number for video generation

	Returns:
		None
	"""

	model.eval()
	for x, y in dataloader:
		break
	generate_singleinput(model, x, target_tensor, 0, count)
	return


class NewVit(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model
	
	def forward(self, x: torch.Tensor):
		# Reshape and permute the input tensor
		x = trained_vit._process_input(x)
		n = x.shape[0]

		# Expand the class token to the full batch
		batch_class_token = trained_vit.class_token.expand(n, -1, -1)
		x = torch.cat([batch_class_token, x], dim=1)

		for i in range(3):
			# x = self.model.transformer.h[i](x)[0]
			x = self.model.encoder.layers[i](x)

		return x

model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2')
trained_vit = torchvision.models.vit_b_16(weights='IMAGENET1K_V1').to(device)
gpt2 = model.to(device)
new_vision = NewVit(trained_vit).to(device)
new_vision.eval()

batch = next(iter(dataloader))
image = batch[0].reshape(1, 3, 224, 224).to(device)
target_tensor = new_vision(image)

target_tensor = target_tensor.detach().to(device)
plt.figure(figsize=(10, 10))
image_width = len(image[0][0])
target_input = image.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
plt.imshow(target_input)
plt.axis('off')
plt.savefig('target_image', bbox_inches='tight', pad_inches=0.1)
plt.close()

modification = torch.randn(1, 3, 224, 224)/15
modification = modification.to(device)
modified_input = image + modification
modified_output = new_vision(modified_input)
print (f'L2 distance between original and shifted inputs: {torch.sqrt(torch.sum((image - modified_input)**2))}')
print (f'L2 distance between target and slightly modified image: {torch.sqrt(torch.sum((target_tensor - modified_output)**2))}')

if __name__ == '__main__':
	new_vision.eval()
	generate_input(dataloader, new_vision, target_tensor, 0)



