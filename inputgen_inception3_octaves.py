# inputgen_inception3_octaves.py
# InceptionV3 applied to image generation using the input gradient descent

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
from prettytable import PrettyTable
from google.colab import files

# files.upload() # upload flower_photos_2
# !unzip flower_photos_2.zip

# dataset directory specification
data_dir = pathlib.Path('flower_photos_2',  fname='Combined')

image_count = len(list(data_dir.glob('*/*.jpg')))
class_names = [item.name for item in data_dir.glob('*') 
			   if item.name not in ['._.DS_Store', '._DS_Store', '.DS_Store']]

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

# specify batch size
minibatch_size = 16
# train_data = ImageDataset(data_dir, image_type='.jpg')
# train_dataloader = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

def clip_image(input_image):
	input_grad = torch.clip(input_image, 1 - 0.1, 0.05)
	return input_image

def save_image(single_input):
	"""
	Saves a .png image of the single_input tensor

	Args:
		single_input: torch.tensor of the input 
		count: int, class number

	Returns:
		None (writes .png to storage)
	"""

	plt.figure(figsize=(10, 10))
	single_input = single_input.clone() / torch.max(single_input)
	image_width = len(single_input[0][0])
	target_input = single_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
	plt.axis('off')
	plt.imshow(target_input, alpha=1)
	plt.tight_layout()
	plt.show()
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


def generate_singleinput(model, input_tensors, output_tensors, count, random_input=True, class_index=558):
	"""
	Generates an input for a given target class

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		count: int, time step
	kwargs:
		random_input: bool, if True then 
		class_index: int, index of Imagenet target class of interest

	returns:
		None (saves .png image)
	"""

	manualSeed = 999
	random.seed(manualSeed)
	torch.manual_seed(manualSeed)

	if random_input:
		single_input = (torch.rand(1, 3, 256, 256))/25 + 0.5 # scaled distribution initialization

	else:
		single_input = input_tensors[0]
 
	single_input = single_input.to(device)
	original_input = torch.clone(single_input).reshape(3, 256, 256).permute(1, 2, 0).cpu().detach().numpy()
	single_input = single_input.reshape(1, 3, 256, 256)
	original_input = torch.clone(single_input).reshape(3, 256, 256).permute(1, 2, 0).cpu().detach().numpy()
	target_output = torch.tensor([class_index], dtype=int)

	for i in range(150):
		single_input = single_input.detach() # remove the gradient for the input (if present)
		input_grad = layer_gradient(Inception3, single_input, target_output) # compute input gradient
		single_input = single_input - 0.1 * input_grad # gradient descent step
		single_input = torchvision.transforms.functional.gaussian_blur(single_input, 11, sigma=2.2-i/80)

	save_image(single_input)
	single_input = torchvision.transforms.Resize([310, 310])(single_input)
	for i in range(100):
		size = 300
		cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
		input_grad = layer_gradient(Inception3, cropped_input, target_output) # compute input gradient
		single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= 0.11 * input_grad # gradient descent step
		single_input = torchvision.transforms.functional.gaussian_blur(single_input, 5, sigma=1.8-i/100)

	save_image(single_input)
	single_input = torchvision.transforms.Resize([340, 340])(single_input)
	for i in range(100):
		size = 320
		cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
		input_grad = layer_gradient(Inception3, cropped_input, target_output) # compute input gradient
		single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= 0.08 * input_grad # gradient descent step
		single_input = torchvision.transforms.functional.gaussian_blur(single_input, 3, sigma=1.5-i/100)

	single_input = torchvision.transforms.Resize([300, 300])(single_input)
	save_image(single_input)
	for i in range(25):
		# single_input = random_crop(single_input)
		single_input = single_input.detach() # remove the gradient for the input (if present)
		input_grad = layer_gradient(model, single_input, target_output) # compute input gradient
		single_input = single_input - 0.03 * input_grad # gradient descent step
		single_input = torchvision.transforms.functional.gaussian_blur(single_input, 3, sigma=0.4)
	
	save_image(single_input)

	return single_input


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
	loss = torch.abs(200 - output[0][int(desired_output)]) 
	loss.backward()
	gradient = input_tensor.grad

	return gradient


def adversarial_test(dataloader, model, count=0):
	model.eval()
	ls = []
	# for x, y in train_dataloader:
	# 	break
	x = []
	for i in range(16):
		ls.append(generate_singleinput(model, x, [], i, count, random_input=True))
	show_batch(ls, grayscale=False)
	x = x.reshape(16, 3, 299, 299).permute(0, 2, 3, 1).cpu().detach().numpy()
	show_batch(x, grayscale=False)
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


class NewModel(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x):
		# N x 3 x 299 x 299
		x = self.model.Conv2d_1a_3x3(x)
		# N x 32 x 149 x 149
		x = self.model.Conv2d_2a_3x3(x)
		# N x 32 x 147 x 147
		x = self.model.Conv2d_2b_3x3(x)
		# N x 64 x 147 x 147
		x = self.model.maxpool1(x)
		# N x 64 x 73 x 73
		x = self.model.Conv2d_3b_1x1(x)
		# N x 80 x 73 x 73
		x = self.model.Conv2d_4a_3x3(x)
		# N x 192 x 71 x 71
		x = self.model.maxpool2(x)
		# N x 192 x 35 x 35
		x = self.model.Mixed_5b(x)
		# N x 256 x 35 x 35
		x = self.model.Mixed_5c(x)
		# N x 288 x 35 x 35
		x = self.model.Mixed_5d(x)
		# N x 288 x 35 x 35
		x = self.model.Mixed_6a(x)
		# N x 768 x 17 x 17
		x = self.model.Mixed_6b(x)
		# N x 768 x 17 x 17
		x = self.model.Mixed_6c(x)
		# N x 768 x 17 x 17
		x = self.model.Mixed_6d(x)
		# N x 768 x 17 x 17
		x = self.model.Mixed_6e(x)
		# N x 768 x 17 x 17
		aux = self.model.AuxLogits(x)
		return aux


loss_fn = nn.CrossEntropyLoss()
Inception3= torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)

if __name__ == '__main__':
	model.eval()
	model.aux_logits = True 
	auxouts = NewModel(Inception3)
	auxouts.eval()
	Inception3.eval()
	adversarial_test(None, auxouts, 0)






