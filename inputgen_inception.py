# inputgen_inception.py
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

# dataset directory specification
data_dir = pathlib.Path('../flower_photos_2',  fname='Combined')

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
		image = torchvision.transforms.Resize(size=[256, 256])(image)	
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
train_data = ImageDataset(data_dir, image_type='.jpg')
train_dataloader = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)
# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")


def transfiguration_video(model, input_tensors, output_tensors, index, count, random_input=True):
	"""
	Generates an input for a given output class

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int
		count: int, timestep
	kwargs:
		random_input: bool, if True then uses a uniform random distribution as the start
		video: bool, if True then saves a .png image for each iteration during generation

	returns:
		None
	"""
	seed = 999
	random.seed(seed)
	torch.manual_seed(seed)

	class_index = 949
	if random_input:
		single_input = (torch.rand(1, 3, 256, 256))/25 + 0.5 # scaled normal distribution initialization

	else:
		single_input = input_tensors[0]

	og_input = torch.clone(single_input)

	for k in range(500):
		single_input = og_input.to(device)
		single_input = single_input.reshape(1, 3, 256, 256)
		original_input = torch.clone(single_input).reshape(3, 256, 256).permute(1, 2, 0).cpu().detach().numpy()
		target_output = torch.tensor([class_index], dtype=int)

		for i in range(100):
			single_input = single_input.detach() # remove the gradient for the input (if present)
			input_grad = layer_gradient(model, single_input, target_output, k) # compute input gradient
			single_input = single_input - 0.15 * input_grad # gradient descent step
			if i < 76:
				single_input = torchvision.transforms.functional.gaussian_blur(single_input, 3)
				if i % 5 == 0:
					single_input = torchvision.transforms.Resize(256)(single_input)
				elif i % 5 == 1:
					single_input = torchvision.transforms.Resize(198)(single_input)
				elif i % 5 == 2:
					single_input = torchvision.transforms.Resize(160)(single_input)
				elif i % 5 == 3:
					single_input = torchvision.transforms.Resize(180)(single_input)
				elif i % 5 == 4:
					single_input = torchvision.transforms.Resize(200)(single_input)

			single_input = torchvision.transforms.ColorJitter(0.001, 0.001, 0.001, 0.001)(single_input)

		single_input = single_input.clone() / torch.max(single_input)
		target_input = single_input.reshape(3, 256, 256).permute(1, 2, 0).cpu().detach().numpy()
		plt.figure(figsize=(18, 10))
		plt.subplot(1, 2, 1)
		plt.axis('off')
		plt.imshow(original_input, alpha=1)
		plt.tight_layout()

		plt.subplot(1, 2, 2)
		plt.axis('off')
		plt.imshow(target_input, alpha=1)
		plt.tight_layout()
		spercent, cpercent = (300-k)/300, k/300
		plt.title(f'{int(spercent*100)}% Strawberry, {int(cpercent*100)}% Castle')
		plt.savefig('adversarial_example{0:04d}.png'.format(k), dpi=410)
		plt.close()

	return target_input

def generate_singleinput(model, input_tensors, output_tensors, index, count, random_input=True, video=False):
	"""
	Generates an input for a given output class

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int
		count: int, timestep
	kwargs:
		random_input: bool, if True then uses a uniform random distribution as the start
		video: bool, if True then saves a .png image for each iteration during generation

	returns:
		None
	"""

	seed = 999
	random.seed(seed)
	torch.manual_seed(seed)

	class_index = 483
	if random_input:
		single_input = (torch.rand(1, 3, 256, 256))/25 + 0.5 # scaled normal distribution initialization

	else:
		single_input = input_tensors[0]

	og_input = torch.clone(single_input)

	single_input = og_input.to(device)
	single_input = single_input.reshape(1, 3, 256, 256)
	original_input = torch.clone(single_input).reshape(3, 256, 256).permute(1, 2, 0).cpu().detach().numpy()
	target_output = torch.tensor([class_index], dtype=int)

	for i in range(100):
		print (i)
		k = 0
		single_input = single_input.detach() # remove the gradient for the input (if present)
		input_grad = layer_gradient(model, single_input, target_output, k, video=True) # compute input gradient
		
		single_input = single_input - 0.15 * input_grad # gradient descent step
		if i < 76:
			single_input = torchvision.transforms.functional.gaussian_blur(single_input, 3)
			if i % 5 == 0:
				single_input = torchvision.transforms.Resize(256)(single_input)
			elif i % 5 == 1:
				single_input = torchvision.transforms.Resize(128)(single_input)
			elif i % 5 == 2:
				single_input = torchvision.transforms.Resize(160)(single_input)
			elif i % 5 == 3:
				single_input = torchvision.transforms.Resize(180)(single_input)
			elif i % 5 == 4:
				single_input = torchvision.transforms.Resize(200)(single_input)
			single_input = torchvision.transforms.Resize(256)(single_input)

		single_input = torchvision.transforms.ColorJitter(0.01, 0.01, 0.01, 0.01)(single_input)

		if video:
			predicted_output = predicted_output.reshape(1000).cpu().detach().numpy()
			color_arr = ['blue' for i in range(len(predicted_output))]
			color_arr[class_index] = 'red'
			color_arr[738] = 'green'

			single_input = single_input.clone() / torch.max(single_input)
			target_input = single_input.reshape(3, 256, 256).permute(1, 2, 0).cpu().detach().numpy()
			plt.figure(figsize=(16, 8))
			plt.subplot(1, 2, 1)
			plt.title('Generated Image')
			plt.imshow(target_input, alpha=1)
			plt.axis('off')
			plt.tight_layout()

			plt.subplot(1, 2, 2)
			plt.title('Predicted Output')
			plt.scatter([i for i in range(1000)], predicted_output, color=color_arr)
			plt.ylim([-10, 80])
			plt.xlabel('ImageNet Category')
			plt.tight_layout()
			plt.savefig('adversarial_example{0:04d}.png'.format(i), dpi=410)
			plt.close()

	return target_input

def generate_inputbatch(model, input_tensors, output_tensors, index, count, minibatch_size, random_input=False):
	"""
	Generates an input for a given output

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int

	returns:
		None (saves .png image)
	"""
	seed = 999
	random.seed(seed)
	torch.manual_seed(seed)

	class_index = 250
	
	for i in range(100):
		if random_input:
			input_tensor = (torch.rand(minibatch_size, 3, 256, 256))/25 + 0.5 # scaled distribution initialization
		else:
			input_tensor = input_tensors

		input_tensor = input_tensor.to(device)
		input_tensor = input_tensor.reshape(minibatch_size, 3, 256, 256)
		original_input = torch.clone(input_tensor).reshape(minibatch_size, 3, 256, 256).permute(0, 2, 3, 1).cpu().detach().numpy()

		target_tensor = torch.zeros(minibatch_size, 1000)
		for j in range(len(target_tensor)):
			target_tensor[j][class_index] = 100 - i
			target_tensor[j][class_index + 1] = i

		for k in range(100):
			input_tensor = input_tensor.detach() # remove the gradient for the input (if present)
			input_grad = target_tensor_gradient(model, input_tensor, target_tensor, minibatch_size) # compute input gradient
			input_tensor = input_tensor - 0.0015 * input_grad # gradient descent step
			if k < 76:
				input_tensor = torchvision.transforms.functional.gaussian_blur(input_tensor, 3)
				if k % 5 == 0:
					input_tensor = torch.nn.functional.interpolate(input_tensor, 256)
				elif k % 5 == 1:
					input_tensor = torch.nn.functional.interpolate(input_tensor, 128)
				elif k % 5 == 2:
					input_tensor = torch.nn.functional.interpolate(input_tensor, 160)
				elif k % 5 == 3:
					input_tensor = torch.nn.functional.interpolate(input_tensor, 100)
				elif k % 5 == 4:
					input_tensor = torch.nn.functional.interpolate(input_tensor, 200)

			input_tensor = torchvision.transforms.ColorJitter(0.01, 0.01, 0.01, 0.01)(input_tensor)
			input_tensor = torch.nn.functional.interpolate(input_tensor, 256)

		input_tensor = input_tensor.reshape(minibatch_size, 3, 256, 256).permute(0, 2, 3, 1).cpu().detach().numpy()
		show_batch(input_tensor, count=i, grayscale=False)

	return input_tensor


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


def layer_gradient(model, input_tensor, desired_output, k, video=False):
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

	if video:
		loss = torch.abs(100 - output[0][int(desired_output)]) * (300-k)/300 + \
			   torch.abs(100 - output[0][int(desired_output) - 466]) * (k/300) + 0.001 * torch.abs(input_tensor).sum() 
	else:
		loss = torch.abs(100 - output[0][int(desired_output)]) + 0.001 * torch.abs(input_tensor).sum() 

	loss.backward()
	gradient = input_tensor.grad
	return gradient


def assemble_inputs(dataloader, model, count=0):
	"""
	Generate inputs 

	Args:
		dataloader: torch.utils.data.Dataloader() object
		model: torch.nn.module object, model of interest
	kwargs:
		count: int, timestep

	Returns:
		None (saves .png image)
	"""

	model.eval()
	ls = []
	# get a minibatch of training examples
	for x, y in train_dataloader:
		break

	for i in range(16):
		ls.append(generate_singleinput(model, x, [], i, count, random_input=False))

	# show generated batch
	show_batch(ls, grayscale=False)
	# show original batch
	x = x.reshape(16, 3, 256, 256).permute(0, 2, 3, 1).cpu().detach().numpy()
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
	plt.close()
	return

if __name__ == '__main__':
	loss_fn = nn.CrossEntropyLoss()
	model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)

	model.eval()
	transfiguration_video()






