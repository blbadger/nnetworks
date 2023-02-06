# feature_visualization.py
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

# dataset directory specification
data_dir = pathlib.Path('flower_photos_3',  fname='Combined')

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
train_dataloader = DataLoader(train_data, batch_size=minibatch_size, shuffle=False)
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



def octave(single_input, target_output, iterations, learning_rates, sigmas, size, index, crop=True):
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
	start_sigma, end_sigma = sigmas

	for i in range(iterations):
		if crop:
			cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
		else:
			cropped_input, crop_height, crop_width = random_crop(single_input.detach(), len(single_input[0][0]))
			size = len(single_input[0][0])
		single_input = single_input.detach() # remove the gradient for the input (if present)
		input_grad = layer_gradient(new_vision, cropped_input, target_output, index) # compute input gradient
		single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)* input_grad # gradient descent step
		single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] = torchvision.transforms.functional.gaussian_blur(single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size], 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations))

	return single_input

def generate_singleinput(model, input_tensors, output_tensors, index, count, random_input=True):
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

	class_index = 13

	if random_input:
		single_input = (torch.rand(1, 3, 224, 224))/10 + 0.5 # scaled  distribution initialization
	else:
		single_input = input_tensors[index]

	single_input = single_input.to(device)
	original_input = torch.clone(single_input).reshape(3, 224, 224).permute(1, 2, 0).cpu().detach().numpy()
	single_input = single_input.reshape(1, 3, 224, 224)
	original_input = torch.clone(single_input).reshape(3, 224, 224).permute(1, 2, 0).cpu().detach().numpy()
	target_output = torch.tensor([class_index], dtype=int)

	single_input = octave(single_input, target_output, 100, [0.5, 0.4], [2.4, 0.8], 0, index, crop=False)
	print ('First Octave Complete')
 
	single_input = torchvision.transforms.Resize([560, 560])(single_input)
	single_input = octave(single_input, target_output, 100, [0.4, 0.3], [1.5, 0.4], 224, index, crop=True)
	print ('Second Octave Complete')

	single_input = torchvision.transforms.Resize([600, 600])(single_input)
	single_input = octave(single_input, target_output, 100, [0.3, 0.2], [1.1, 0.3], 224, index, crop=True)
	print ('Third Octave Complete')

	image_dim = len(single_input[0][0])
	target_input = single_input.reshape(3, image_dim, image_dim).permute(1, 2, 0).cpu().detach().numpy()
	# plt.figure(figsize=(15, 10))
	# plt.subplot(1, 2, 1)
	plt.style.use('dark_background')
	plt.axis('off')
	plt.imshow(target_input, alpha=1)
	plt.tight_layout()
	plt.savefig('feature_vis{0:04d}.png'.format(index), dpi=310, pad_inches=0, bbox_inches=0)

	# plt.subplot(1, 2, 2)
	# plt.axis('off')
	# plt.imshow(target_input, alpha=1)
	# plt.tight_layout()
	# plt.savefig('feature_viz{0:04d}.png'.format(count), dpi=410)
	# plt.close()

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
	output = new_vision(input_tensor)
	loss = 0.5*(200 - output[0][int(desired_output)]) # no L1 regularization needed
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
	focus = output[:, :, index]
	target = torch.ones(focus.shape).to(device)*200
	loss = 20*torch.sum(target - focus)
	loss.backward()
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
	x = []
	for x, y in train_dataloader:
		break
	for i in range(8):
		ls.append(generate_singleinput(model, x, [], i, count, random_input=True))
	show_batch(ls, grayscale=False)
	# observe the original images
	# x = x.reshape(16, 3, 299, 299).permute(0, 2, 3, 1).cpu().detach().numpy()
	# show_batch(x, grayscale=False)
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

	plt.figure(figsize=(15, 8))
	for n in range(8):
		ax = plt.subplot(2, 4, n+1)
		plt.axis('off')
		if grayscale:
			plt.imshow(input_batch[n], cmap='gray')
		else:
			plt.imshow(input_batch[n])
		plt.tight_layout()

	plt.tight_layout()
	plt.savefig('vit_feature{0:04d}.png'.format(count), dpi=360, transparent=True)
	plt.close()
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
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        for i in range(3):
            x = self.model.encoder.layers[i](x)

        inp = x
       	x = self.model.encoder.layers[i+1].ln_1(x)
        x, _ = self.model.encoder.layers[i+1].self_attention(x, x, x, need_weights=False)
        x = x + inp

        y = self.model.encoder.layers[i+1].ln_2(x)
        y = self.model.encoder.layers[i+1].mlp(y)
        x = x + y

        # # Classifier "token" as used by standard language architectures
        # x = x[:, 0]

        # x = self.model.heads(x)
        return x 

trained_vit = torchvision.models.vit_b_16(weights='IMAGENET1K_V1').to(device)
untrained_vit = torchvision.models.vit_b_16().to(device)
new_vision = NewVit(trained_vit).to(device)
new_vision.eval()

if __name__ == '__main__':
	new_vision.eval()
	generate_dream(None, new_vision, 0)