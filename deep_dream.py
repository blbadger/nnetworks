# deep_dream.py
# A deep convolutional net for image classification
# implemented with a functional pytorch model

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
# from google.colab import files

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

	def __init__(self, img_dir, transform=None, target_transform=None, image_type='.png', image_size=299):
		# specify image labels by folder name 
		self.img_labels = [item.name for item in data_dir.glob('*')]

		# construct image name list: randomly sample images for each epoch
		images = list(img_dir.glob('*/*' + image_type))
		random.shuffle(images)
		self.image_name_ls = images[:800]

		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform
		self.image_size = image_size

	def __len__(self):
		return len(self.image_name_ls)

	def __getitem__(self, index):
		# path to image
		img_path = os.path.join(self.image_name_ls[index])
		image = torchvision.io.read_image(img_path) # convert image to tensor of ints
		image = image / 255. # convert ints to floats in range [0, 1]
		image = torchvision.transforms.Resize(size=[self.image_size, self.image_size])(image)

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
train_data = ImageDataset(data_dir, image_type='.jpg', image_size=500)
# train_dataloader = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)
# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")


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
	plt.show()
	# images_dir = '/content/gdrive/My Drive/googlenet_generation_nopad'
	# plt.savefig("{}".format(images_dir) + "/Class {0:04d}- ".format(count) + "{}.png".format(imagenet_classes[count]), bbox_inches='tight', pad_inches=0, dpi=390)
	plt.close()
	return

def octave(single_input, target_output, iterations, learning_rates, sigmas, size, index, crop=True, blur=True):
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
		input_grad = layer_gradient(newmodel, cropped_input, target_output, index) # compute input gradient
		single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)* input_grad # gradient descent step
		if blur:
			single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] = torchvision.transforms.functional.gaussian_blur(single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size], 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations))

	return single_input


def generate_singleinput(model, input_tensors, output_tensors, index, count, random_input=True):
	"""
	Generates an input for a given output category

	Args:
		input_tensors: torch.Tensor object, minibatch of inputs
		output_tensors: torch.Tensor object, minibatch of outputs
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

	if random_input:
		single_input = (torch.rand(1, 3, 299, 299))/10 + 0.5 # scaled  distribution initialization
	else:
		single_input = input_tensors[index]

	single_input = single_input.to(device)
	original_input = torch.clone(single_input).reshape(3, 299, 299).permute(1, 2, 0).cpu().detach().numpy()
	single_input = single_input.reshape(1, 3, 299, 299)
	original_input = torch.clone(single_input).reshape(3, 299, 299).permute(1, 2, 0).cpu().detach().numpy()
	target_output = torch.tensor([class_index], dtype=int)

	single_input = octave(single_input, target_output, 100, [0.5, 0.4], [2.4, 0.8], 0, index, crop=False)

	single_input = torchvision.transforms.Resize([380, 380])(single_input)
	single_input = octave(single_input, target_output, 100, [0.4, 0.3], [1.5, 0.4], 330, index, crop=False)

	single_input = torchvision.transforms.Resize([460, 460])(single_input)
	single_input = octave(single_input, target_output, 100, [0.3, 0.2], [1.1, 0.3], 375, index, crop=False)

	image_dim = len(single_input[0][0])
	target_input = single_input.reshape(3, image_dim, image_dim).permute(1, 2, 0).cpu().detach().numpy()
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
	# plt.close()

	return target_input



def generate_highres_dream(model, input_tensors, output_tensors, index, count):
	"""
	Generates a high resolution deep dream image. Expects an input
	of size 500x500

	Args:
		input_tensors: torch.Tensor object
		output_tensors: torch.Tensor object
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

	single_input = input_tensors[index].to(device)
	original_input = torch.clone(single_input).reshape(3, 500, 500).permute(1, 2, 0).cpu().detach().numpy()
	single_input = single_input.reshape(1, 3, 500, 500)
	original_input = torch.clone(single_input).reshape(3, 500, 500).permute(1, 2, 0).cpu().detach().numpy()
	target_output = torch.tensor([class_index], dtype=int)
	single_input = octave(single_input, target_output, 100, [0.5, 0.4], [2.4, 0.8], 0, index, crop=False)
 
	single_input = torchvision.transforms.Resize([600, 600])(single_input)
	single_input = octave(single_input, target_output, 100, [0.4, 0.3], [1.5, 0.4], 330, index, crop=False)

	single_input = torchvision.transforms.Resize([900, 900])(single_input)
	single_input = octave(single_input, target_output, 100, [0.3, 0.2], [1.1, 0.3], 375, index, crop=False)

	image_dim = len(single_input[0][0])
	target_input = single_input.reshape(3, image_dim, image_dim).permute(1, 2, 0).cpu().detach().numpy()
	plt.figure(figsize=(20, 15))
	plt.subplot(1, 2, 1)
	plt.axis('off')
	plt.imshow(original_input, alpha=1)
	plt.tight_layout()

	plt.subplot(1, 2, 2)
	plt.axis('off')
	plt.imshow(target_input, alpha=1)
	plt.tight_layout()
	plt.show()

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


def target_tensor_gradient(model, input_tensor, desired_output, minibatch_size):
	"""
	Compute the gradient of the output (logits) with respect to the input 
	using an L1-normalized L1 metric to maximize the target classification.

	Args:
		model: torch.nn.model
		input_tensor: torch.tensor object corresponding to the input image
		true_output: torch.tensor object of the desired classification label

	Returns:
		gradient: torch.tensor.grad on the input tensor after backpropegation

	"""

	input_tensor.requires_grad = True
	output = model(input_tensor)
	loss = torch.sum(desired_output - output) + 0.001 * torch.abs(input_tensor).sum() 
	loss.backward()
	gradient = input_tensor.grad

	return gradient


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
	output = model(input_tensor).to(device)
	focus = output[0][:][:][:]
	target = torch.ones(focus.shape).to(device)*200
	loss = 0.001*torch.sum(target - focus) #0.002 for 6b
	loss.backward()
	gradient = input_tensor.grad

	return gradient

def double_layer_gradient(model, input_tensor, desired_output, index):
	"""
	Compute the gradient of two layers w.r.t. the input simultaneously

	Args:
		model: torch.nn.model

	Returns:
		gradient: torch.tensor.grad on the input tensor after backpropegation
	"""

	input_tensor.requires_grad = True
	output = model(input_tensor).to(device)
	output2 = newmodel2(input_tensor).to(device)

	focus = output[0][:][:][:]
	focus2 = output2[0][:][:][:]

	target = torch.ones(focus.shape).to(device)*200
	target2 = torch.ones(focus2.shape).to(device)*200

	output = model(input_tensor)
	loss = torch.sum(torch.abs(output))
	loss = 0.0001*(torch.sum(target - focus) + torch.sum(target2 - focus2))
	loss.backward() # back-propegate loss
	gradient = input_tensor.grad

	return gradient


def generate_dream(dataloader, model, count=0):
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
	ls = []

	for x, y in train_dataloader:
		break

	for i in range(16):
		ls.append(generate_singleinput(model, x, [], i, count, random_input=False))

	show_batch(ls, grayscale=False)
	# observe the original images
	x = x.reshape(16, 3, 299, 299).permute(0, 2, 3, 1).cpu().detach().numpy()
	show_batch(x, grayscale=False)

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
	plt.show()
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
		# x = self.model.Mixed_6b(x)
		# N x 768 x 17 x 17
		# x = self.model.Mixed_6c(x)
		# N x 768 x 17 x 17
		# x = self.model.Mixed_6d(x)
		# N x 768 x 17 x 17
		# x = self.model.Mixed_6e(x)
		# N x 768 x 17 x 17
		# aux = self.model.AuxLogits(x)
		# N x 768 x 17 x 17
		# x = self.model.Mixed_7a(x)
		# N x 1280 x 8 x 8
		# x = self.model.Mixed_7b(x)
		# N x 2048 x 8 x 8
		# x = self.model.Mixed_7c(x)
		return x

class NewModel2(nn.Module):

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
		# x = self.model.maxpool1(x)
		# N x 64 x 73 x 73
		# x = self.model.Conv2d_3b_1x1(x)
		# N x 80 x 73 x 73
		# x = self.model.Conv2d_4a_3x3(x)
		# N x 192 x 71 x 71
		# x = self.model.maxpool2(x)
		# N x 192 x 35 x 35
		# x = self.model.Mixed_5b(x)
		# N x 256 x 35 x 35
		# x = self.model.Mixed_5c(x)
		# N x 288 x 35 x 35
		# x = self.model.Mixed_5d(x)
		# N x 288 x 35 x 35
		# x = self.model.Mixed_6a(x)
		# N x 768 x 17 x 17 	
		# x = self.model.Mixed_6b(x)
		# N x 768 x 17 x 17
		# x = self.model.Mixed_6c(x)
		# N x 768 x 17 x 17
		# x = self.model.Mixed_6d(x)
		return x

class NewModel3(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x):
			# N x 3 x 224 x 224
			x = self.model.conv1(x)
			# N x 64 x 112 x 112
			x = self.model.maxpool1(x)
			# N x 64 x 56 x 56
			x = self.model.conv2(x)
			# N x 64 x 56 x 56
			# x = self.model.conv3(x)
			# N x 192 x 56 x 56
			# x = self.model.maxpool2(x)
			# N x 192 x 28 x 28
			# x = self.model.inception3a(x)
			# N x 256 x 28 x 28
			# x = self.model.inception3b(x)
			# N x 480 x 28 x 28
			# x = self.model.maxpool3(x)
			# N x 480 x 14 x 14
			# x = self.model.inception4a(x)
			# N x 512 x 14 x 14
			# x = self.model.inception4b(x)
			# N x 512 x 14 x 14
			# x = self.model.inception4c(x)
			# # N x 512 x 14 x 14
			# x = self.model.inception4d(x)
			# # N x 528 x 14 x 14
			# x = self.model.inception4e(x)
			# # N x 832 x 14 x 14
			# x = self.model.maxpool4(x)
			# # N x 832 x 7 x 7
			# x = self.model.inception5a(x)
			# # N x 832 x 7 x 7
			# x = self.model.inception5b(x)
			# # N x 1024 x 7 x 7
			# x = self.model.avgpool(x)
			# # N x 1024 x 1 x 1
			# x = torch.flatten(x, 1)
			# # N x 1024
			# x = self.model.dropout(x)
			# x = self.model.fc(x)
			# N x 1000 (num_classes)
			return x

class NewModel4(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x):
		x = self.model.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)

		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)
		# x = self.model.layer4(x)

		# x = self.model.avgpool(x)
		# x = torch.model.flatten(x, 1)
		# x = self.model.fc(x)
		return x


loss_fn = nn.CrossEntropyLoss()
Inception3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)
googlenet = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True).to(device)
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)
count_parameters(Inception3)

if __name__ == '__main__':
	newmodel = NewModel(Inception3)
	# newmodel2 = NewModel2(Inception3)
	# newmodel = NewModel3(googlenet)
	# newmodel = NewModel4(resnet)
	newmodel.eval()
	generate_dream(None, newmodel, 0)


