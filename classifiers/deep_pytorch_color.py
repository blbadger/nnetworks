# deep_pytorch.py
# A deep convolutional net for image classification, input attribution, and image generation.
# Optimized for low-dimensional outputs.

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

# dataset directory specification
data_dir = pathlib.Path('../Neural_networks/flower_1',  fname='Combined')
data_dir2 = pathlib.Path('../Neural_networks/flower_2', fname='Combined')
data_dir3 = pathlib.Path('../Neural_networks/flower_2', fname='Combined') # nnetworks_data/file

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

		# construct image name list: randomly sample 400 images for each epoch
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
		image = torchvision.io.read_image(img_path) # convert image to tensor of ints
		image = image / 255. # convert ints to floats in range [0, 1]
		image = torchvision.transforms.Resize(size=[256, 256])(image)	

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
batch_size = 16

train_data = ImageDataset(data_dir, image_type='.jpg')
test_data = ImageDataset(data_dir2, image_type='.jpg')
test_data2 = ImageDataset(data_dir3, image_type='.jpg')

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class DeepNetwork(nn.Module):

	def __init__(self):

		super(DeepNetwork, self).__init__()
		self.entry_conv = Conv2d(3, 16, 3, padding=(1, 1))
		self.conv16 = Conv2d(16, 16, 3, padding=(1, 1))
		self.conv32 = Conv2d(16, 32, 3, padding=(1, 1))
		self.conv32_2 = Conv2d(32, 32, 3, padding=(1, 1))
		self.conv64 = Conv2d(32, 64, 3, padding=(1, 1))
		self.conv64_2 = Conv2d(64, 64, 3, padding=(1, 1))

		self.max_pooling = nn.MaxPool2d(2)
		self.flatten = nn.Flatten()
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)
		self.d1 = nn.Linear(8192, 512)
		self.d2 = nn.Linear(512, 50)
		self.d3 = nn.Linear(50, 2)
		

	def forward(self, model_input):
		out = self.relu(self.entry_conv(model_input))
		out = self.max_pooling(out)
		out = self.relu(self.conv16(out))
		out = self.max_pooling(out)
		out = self.relu(self.conv16(out))
		out = self.max_pooling(out)

		out = self.relu(self.conv32(out))
		out = self.max_pooling(out)
		out = self.relu(self.conv32_2(out))
		out = self.relu(self.conv32_2(out))
		out = self.max_pooling(out)
		out = self.relu(self.conv64(out))
		out = self.max_pooling(out)
		out = self.relu(self.conv64_2(out))
		out = self.max_pooling(out)
		out = self.relu(self.conv64_2(out))
		out = self.max_pooling(out)
		output = torch.flatten(out, 1, 3)

		output = self.d1(output)
		output = self.relu(output)
		output = self.d2(output)
		output = self.relu(output)
		final_output = self.d3(output)
		final_output = self.softmax(final_output)
		return final_output

class MediumNetwork(nn.Module):

	def __init__(self):

		super(MediumNetwork, self).__init__()
		self.entry_conv = Conv2d(3, 16, 3, padding=(1, 1))
		self.conv16 = Conv2d(16, 16, 3, padding=(1, 1))
		self.conv32 = Conv2d(16, 32, 3, padding=(1, 1))
		self.conv32_2 = Conv2d(32, 32, 3, padding=(1, 1))
		self.conv64 = Conv2d(32, 64, 3, padding=(1, 1))
		self.conv64_2 = Conv2d(64, 64, 3, padding=(1, 1))

		self.max_pooling = nn.MaxPool2d(2)
		self.flatten = nn.Flatten()
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)
		self.sigmoid = nn.Sigmoid()
		self.d1 = nn.Linear(8192, 512)
		self.d2 = nn.Linear(512, 50)
		self.d3 = nn.Linear(50, 5)
		

	def forward(self, model_input):
		out = self.relu(self.entry_conv(model_input))
		out = self.max_pooling(out)
		out = self.relu(self.conv16(out))
		out = self.max_pooling(out)
		out = self.relu(self.conv16(out))
		out = self.max_pooling(out)
		out = self.relu(self.conv32(out))
		out = self.max_pooling(out)
		output = torch.flatten(out, 1, 3)

		output = self.d1(output)
		output = self.relu(output)
		output = self.d2(output)
		output = self.relu(output)
		final_output = self.d3(output)
		final_output = self.sigmoid(final_output)
		return final_output


def gradientxinput(model, input_tensor, output_dim, max_normalized=False):
	"""
	 Compute a gradientxinput attribution score

	 Args:
		input: torch.Tensor() object of input
		model: Transformer() class object, trained neural network

	 Returns:
		gradientxinput: arr[float] of input attributions

	"""

	# change output to float
	input_tensor.requires_grad = True
	output = model.forward(input_tensor)

	# only scalars may be assigned a gradient
	output = output.reshape(1, output_dim).max()

	# backpropegate output gradient to input
	output.backward(retain_graph=True)
	gradientxinput = torch.abs(input_tensor.grad) * input_tensor

	if max_normalized:
		max_val = torch.max(gradientxinput)
		if max_val > 0:
			gradientxinput = gradientxinput / max_val

	return gradientxinput

def hidden_gradient(model, input_tensor, output_dim, max_normalized=False):
	"""
	 Compute a gradientxinput attribution score

	 Args:
		input: torch.Tensor() object of input
		model: Transformer() class object, trained neural network

	 Returns:
		gradientxinput: arr[float] of input attributions

	"""

	# change output to float
	input_tensor.requires_grad = True
	output = model.forward(input_tensor)

	# only scalars may be assigned a gradient
	hidden_activation = model.conv64_2.bias[0]
	hidden_activation += 0.1

	# backpropegate output gradient to input
	hidden_activation.backward(retain_graph=True)
	gradient = input_tensor.grad

	return gradient

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


def loss_gradientxinput(model, input_tensor, true_output, output_dim, max_normalized=False):
	"""
	 Compute a gradientxinput attribution score

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

	# only scalars may be assigned a gradient
	output = output.reshape(1, output_dim).max()

	# backpropegate output gradient to input
	loss.backward(retain_graph=True)
	gradientxinput = torch.abs(input_tensor.grad) * input_tensor

	if max_normalized:
		max_val = torch.max(gradientxinput)
		if max_val > 0:
			gradientxinput = gradientxinput / max_val

	return gradientxinput


def show_batch(input_batch, output_batch, gradxinput_batch, individuals=False, count=0):
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
	if individuals:
		for n in range(len(input_batch)):
			ax = plt.subplot(1, 3, 1)
			plt.axis('off')
			plt.title('Input')
			plt.imshow(input_batch[n], alpha=1)
			ax = plt.subplot(1, 3, 2)
			plt.axis('off')
			plt.title('Gradient * Input')
			plt.imshow(gradxinput_batch[n], cmap='inferno', alpha=1)

			ax = plt.subplot(1, 3, 3)
			plt.axis('off')
			plt.title('Combined')
			plt.imshow(input_batch[n], alpha=1)
			plt.imshow(gradxinput_batch[n], cmap='inferno', alpha=0.5)
			plt.tight_layout()
			plt.savefig(f'attribution{n}.png', dpi=410)
			plt.close()

	plt.figure(figsize=(15, 10))
	for n in range(len(input_batch)):
		ax = plt.subplot(4, 4, n+1) # expects a batch of size 16
		plt.axis('off')
		# plt.title(class_names[int(output_batch[n])].title())
		plt.imshow(input_batch[n], alpha=1)
		# plt.imshow(gradxinput_batch[n], cmap='inferno', alpha=1)

	plt.tight_layout()
	plt.savefig('flower_attributions{0:04d}.png'.format(count), dpi=410)
	plt.close()
	return

def plot_adversaries(model, input_tensors, output_tensors, index, count):
	"""
	Plots adversarial examples by applying the gradient of the loss with respect to the input.

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int, example number
		count: int, timestep number

	returns:
		None (saves .png image)
	"""

	single_input = input_tensors[index].reshape(1, 3, 256, 256)
	input_grad = loss_gradient(model, single_input, output_tensors[index], 5)
	input_grad = torch.rand(input_grad.shape)
	input_grad /= torch.max(input_grad)
	added_input = single_input + 0.1*input_grad
	original_pred = model(single_input)
	grad_pred = model(0.01*input_grad)
	adversarial_pred = model(added_input)

	input_img = single_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
	gradient = input_grad.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
	added_input = added_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()

	original_class = class_names[int(original_pred.argmax(1))].title()
	original_confidence = int(max(original_pred.detach().numpy()[0]) * 100)
	actual_class = class_names[int(output_tensors[index])].title()

	adversarial_class = class_names[int(adversarial_pred.argmax(1))].title()
	adversarial_confidence = int(max(adversarial_pred.detach().numpy()[0]) * 100)

	grad_class = class_names[int(grad_pred.argmax(1))].title() 
	grad_confidence = int(max(grad_pred.detach().numpy()[0]) * 100)

	ax = plt.subplot(1, 3, 1)
	plt.axis('off')
	plt.title('{}% {}'.format(original_confidence, original_class))
	plt.imshow(input_img, alpha=1)
	ax = plt.subplot(1, 3, 2)
	plt.axis('off')
	plt.title('Random Addition')
	plt.imshow(gradient, alpha=1)

	ax = plt.subplot(1, 3, 3)
	plt.axis('off')
	plt.title('{}% {}'.format(adversarial_confidence, adversarial_class))
	plt.imshow(added_input, alpha=1)
	plt.tight_layout()
	plt.savefig('adversarial_example{0:04d}.png'.format(count), dpi=410)
	plt.close()

	return

def generate_adversaries(model, input_tensors, output_tensors, index, count):
	"""
	Plots adversarial examples by applying the gradient of the loss with respect to the input.

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int, example number
		count: timestep number

	returns:
		None (saves .png image)
	"""

	single_input = input_tensors[index].reshape(1, 3, 256, 256)
	input_grad = loss_gradient(model, single_input, output_tensors[index], 5)
	input_grad /= torch.max(input_grad)
	added_input = single_input + 0.15*input_grad
	original_pred = model(single_input)
	grad_pred = model(0.01*input_grad)

	input_img = single_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
	gradient = 15*input_grad.res
	gradient = gradient.shape(3, 256, 256).permute(1, 2, 0).detach().numpy()

	plt.figure(figsize=(18, 10))
	ax = plt.subplot(1, 2, 1)
	plt.axis('off')
	plt.imshow(input_img, alpha=1)
	ax = plt.subplot(1, 2, 2)
	plt.axis('off')
	plt.imshow(gradient, alpha=1)
	plt.tight_layout()
	plt.savefig('adversarial_example{0:04d}.png'.format(count), dpi=410)
	plt.close()

	return

def hidden_input_gen(model, input_tensors, output_tensors, index, count):
	"""
	Generates an input for a given output

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int, example number
		count: int, timestep number

	returns:
		None (saves .png image)
	"""

	target_input = input_tensors[index].reshape(1, 3, 256, 256)
	single_input = torch.rand(1, 3, 256, 256) # uniform distribution initialization
	output_tensors[index] = torch.Tensor([4])

	for i in range(1000):
		single_input = single_input.detach() # remove the gradient for the input (if present)
		input_grad = hidden_gradient(model, single_input, output_tensors[index], 5) # compute input gradient
		last_input = single_input.clone()
		single_input = single_input - 10*input_grad # gradient descent step

	single_input = single_input.clone() / torch.max(single_input) * 10
	single_input = single_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
	target_input = target_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
	target_name = class_names[int(output_tensors[index])].title()

	plt.axis('off')
	plt.title(f'{target_name}')
	plt.imshow(single_input, alpha=1)
	plt.savefig('adversarial_example{0:04d}.png'.format(count), dpi=410)
	plt.close()

	return

def generate_input(model, input_tensors, output_tensors, index, count):
	"""
	Generates an input for a given output

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int, example number
		count: int, timestep number

	returns:
		None (saves .png image)
	"""

	original_input = input_tensors[index].reshape(1, 3, 256, 256)
	single_input = original_input.clone()
	target_output = torch.tensor([0], dtype=int)
	# single_input = torch.zeros(1, 3, 256, 256) # uniform distribution initialization

	for i in range(100):
		single_input = single_input.detach() # remove the gradient for the input (if present)
		# predicted_output = output_tensors[index].argmax(0)
		input_grad = loss_gradient(model, single_input, target_output, 5) # compute input gradient
		single_input = single_input - 0.01*torch.sign(input_grad) # gradient descent step

	single_input = single_input.clone() / torch.max(single_input) 
	single_input = single_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
	original_input = original_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
	target_name = class_names[int(target_output)].title()
	original_name = class_names[int(output_tensors[index])].title()

	plt.figure(figsize=(9, 5))
	plt.subplot(1, 2, 1)
	plt.axis('off')
	plt.title(f'{original_name}')
	plt.imshow(original_input, alpha=1)

	plt.subplot(1, 2, 2)
	plt.axis('off')
	plt.title(f'{target_name}')
	plt.imshow(single_input, alpha=1)
	plt.savefig('generated_input{0:04d}.png'.format(count), dpi=300)
	plt.close()

	return

def train(dataloader, model, loss_fn, optimizer, epochs):
	"""
	Train the model on a dataset from the dataloader object

	Args:
		dataloader: torch.utis.Dataloader() object, iterable for loading transformed data
		model: torch.nn.module() object, deep learning model of interest
		loss_fn: arbitrary fn, loss function for application to model outputs
		optimizer: torch.nn object for performing gradient descent parameter updates
		epochs: int, number of training epochs desired

	Returns:
		None (updates model in-place)
	"""

	model.train()
	count = 0
	total_loss = 0
	start = time.time()

	for e in range(epochs):
		print (f"Epoch {e+1} \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		print ('\n')
		for batch, (x, y) in enumerate(dataloader):
			count += 1
			x, y = x.to(device), y.to(device)
			pred = model(x)
			loss = loss_fn(pred, y)
			total_loss += loss

			# zero out gradients to prevent addition between minibatches
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		ave_loss = float(total_loss) / count
		elapsed_time = time.time() - start
		print (f"Average Loss: {ave_loss:.04}")
		print (f"Completed in {int(elapsed_time)} seconds")
		start = time.time()
	return

def adversarial_test(dataloader, model, count=0):
	"""
	Train the model on a dataset from the dataloader object

	Args:
		dataloader: torch.utis.Dataloader() object, iterable for loading transformed data
		model: torch.nn.module() object, deep learning model of interest
	kwargs:
		count: int, timestep number

	Returns:
		None (updates model in-place)
	"""
	size = len(dataloader.dataset)	
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for x, y in dataloader:
			x, y = x.to(device), y.to(device)
			break

	inputs, gradxinputs = [], []
	for i in range(10):
		generate_input(model, x, y, i, count=i)
		plot_adversaries(model, x, y, i, count=i)
	model.train()
	return

def test(dataloader, model, count=0, short=True):
	"""
	Train the model on a dataset from the dataloader object

	Args:
		dataloader: torch.utis.Dataloader() object, iterable for loading transformed data
		model: torch.nn.module() object, deep learning model of interest
	kwargs:
		count: int, timestep number
		short: bool, if true then only one minibatch is tested

	Returns:
		None (updates model in-place)
	"""
	size = len(dataloader.dataset)	
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for x, y in dataloader:
			x, y = x.to(device), y.to(device)
			pred = model(x)
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
			if short:
				break

	inputs, gradxinputs = [], []
	for i in range(len(x[:6])):
		single_input= x[i].reshape(1, 3, 256, 256)
		gxi = loss_gradient(model, single_input, y[i], 5)
		# gxi = torch.sum(gxi, 1)
		input_img = single_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
		gxi_img = gxi / torch.max(gxi) * 10
		gxi_img = gxi_img.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
		inputs.append(input_img)
		gradxinputs.append(gxi_img)

	show_batch(inputs, y, gradxinputs, count=count)
	accuracy = correct / size
	print (f"Test accuracy: {int(correct)} / {size}")
	model.train()
	return


if __name__ == '__main__':
	training = False
	epochs = 40
	model = MediumNetwork()
	loss_fn = nn.CrossEntropyLoss() 
	if training:
		train(train_dataloader, model, loss_fn, optimizer, epochs)
		torch.save(model.state_dict(), 'trained_models/flowernet.pth')

	model.load_state_dict(torch.load('trained_models/flowernet.pth'))
	test(test_dataloader, model)
	adversarial_test(test_dataloader, model, 0)




