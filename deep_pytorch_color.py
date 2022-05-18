# deep_pytorch.py
# A deep convolutional net for image classification
# implemented with a functional pytorch model

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

class MediumNetworkFull(nn.Module):

	def __init__(self):

		super(MediumNetworkFull, self).__init__()
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

class MediumNetwork(nn.Module):

	def __init__(self):

		super(MediumNetwork, self).__init__()
		self.entry_conv = Conv2d(3, 16, 3, padding=(1, 1))
		self.conv16 = Conv2d(16, 16, 3, padding=(1, 1))
		self.conv32 = Conv2d(16, 32, 3, padding=(1, 1))
		self.conv32_2 = Conv2d(32, 32, 3, padding=(1, 1))
		self.conv64 = Conv2d(32, 64, 3, padding=(1, 1))
		self.conv64_2 = Conv2d(64, 64, 3, padding=(1, 1))

		self.max_pooling = nn.MaxPool2d(2, return_indices=True)
		self.flatten = nn.Flatten()
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)
		self.d1 = nn.Linear(8192, 512)
		self.d2 = nn.Linear(512, 50)
		self.d3 = nn.Linear(50, 2)
		self.index1, self.index2, self.index3, self.index4 = [], [], [], []
		self.batchnorm1 = nn.BatchNorm1d(512)
		self.batchnorm2 = nn.BatchNorm1d(50)
		

	def forward(self, model_input):
		out = self.relu(self.entry_conv(model_input))
		out, self.index1 = self.max_pooling(out)
		out = self.relu(self.conv16(out))
		out, self.index2 = self.max_pooling(out)
		out = self.relu(self.conv16(out))
		out, self.index3 = self.max_pooling(out)
		out = self.relu(self.conv32(out))
		out, self.index4 = self.max_pooling(out)
		output = torch.flatten(out, 1, 3)

		output = self.d1(output)
		output = self.relu(output)
		output = self.d2(output)
		output = self.relu(output)
		final_output = self.d3(output)
		final_output = self.softmax(final_output)
		return final_output


class InvertedMediumNet(nn.Module):

	def __init__(self):

		super(InvertedMediumNet, self).__init__()
		self.entry_conv = Conv2d(16, 3, 3, padding=(1, 1))
		self.conv16 = Conv2d(16, 16, 3, padding=(1, 1))
		self.conv32 = Conv2d(32, 16, 3, padding=(1, 1))
		self.conv32_2 = Conv2d(32, 32, 3, padding=(1, 1))
		self.conv64 = Conv2d(64, 32, 3, padding=(1, 1))
		self.conv64_2 = Conv2d(64, 64, 3, padding=(1, 1))

		self.max_pooling = nn.MaxUnpool2d(2)
		self.flatten = nn.Flatten()
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)
		self.d1 = nn.Linear(512, 8192)
		self.d2 = nn.Linear(50, 512)
		self.d3 = nn.Linear(5, 50)
		self.batchnorm1 = nn.BatchNorm1d(512)
		self.batchnorm2 = nn.BatchNorm1d(50)
		

	def forward(self, final_output):
		# final_output = self.softmax(final_output)
		output = self.d3(final_output)
		output = self.relu(output)
		output = self.d2(output)
		output = self.relu(output)
		output = self.d1(output)

		out = torch.reshape(output, (16, 32, 16, 16)) # reshape for convolutions
		out = self.max_pooling(out, discriminator.index4)
	
		out = self.relu(self.conv32(out))
		out = self.max_pooling(out, discriminator.index3)
		out = self.relu(self.conv16(out))
		out = self.max_pooling(out, discriminator.index2)
		out = self.relu(self.conv16(out))
		out = self.max_pooling(out, discriminator.index1)
		out = self.relu(self.entry_conv(out))
		return out


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

def train(dataloader, model, loss_fn, optimizer, epochs):
	model.train()
	count = 0
	total_loss = 0
	start = time.time()

	for e in range(epochs):
		print (f"Epoch {e+1} \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		print ('\n')
		for batch, (x, y) in enumerate(dataloader):
			print (x.shape)
			# test(test_dataloader, model, count)
			# adversarial_test(test_dataloader, model, count)
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


def train_generative_adversaries(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs):
	discriminator.train()
	generator.train()
	count = 0
	total_loss = 0
	start = time.time()

	for e in range(epochs):
		print (f"Epoch {e+1} \n~~~~~~~~~~~~~~~~~~~~~~~")
		print ('\n')
		for batch, (x, y) in enumerate(dataloader):
			count += 1

			discriminator_optimizer.zero_grad()
			_ = discriminator(x) # initialize the index arrays

			random_output = torch.randn(16, 5)
			generated_samples = generator(random_output)
			# input_dataset = torch.cat([x, generated_samples]) # or torch.cat([x, torch.rand(x.shape)])
			# output_labels = torch.cat([torch.ones(len(y), dtype=int), torch.zeros(len(generated_samples), dtype=int)])
			discriminator_prediction = discriminator(generated_samples)
			output_labels = torch.zeros(len(y), dtype=int) # generated examples have label 0
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)
			discriminator_loss.backward()

			discriminator_prediction = discriminator(x)
			output_labels = torch.ones(len(y), dtype=int) # true examples have label 1
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)
			discriminator_loss.backward()
			discriminator_optimizer.step()
			print (f'Discriminator loss: {discriminator_loss}')
			print (discriminator.d2.bias.norm())

			_ = discriminator(x) # reset index dims to 16-element minibatch size
				
			generated_outputs = generator(random_output)
			discriminator_outputs = discriminator(generated_outputs)
			generator_loss = loss_fn(discriminator_outputs, torch.ones(len(y), dtype=int)) # pretend that all generated inputs are in the dataset

			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()

			print (generator.d2.bias.norm())
			print (discriminator_prediction)
			print (generator_loss)
			plt.axis('off')
			output = generator(random_output)
			plt.imshow((generated_outputs[0] / torch.max(generated_outputs[0])).reshape(3, 256, 256).permute(1, 2, 0).detach().numpy())
			plt.tight_layout()
			plt.savefig('gan{0:04d}'.format(count), dpi=410)
			plt.close()

		torch.save(generator.state_dict(), 'trained_models/generator_large.pth')
		torch.save(discriminator.state_dict(), 'trained_models/discriminator_large.pth')
		ave_loss = float(total_loss) / count
		elapsed_time = time.time() - start
		print (f"Average Loss: {ave_loss:.04}")
		print (f"Completed in {int(elapsed_time)} seconds")
		start = time.time()

	return

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
	# plt.show()
	plt.savefig('flower_attributions{0:04d}.png'.format(count), dpi=410)
	plt.close()
	return

def plot_adversaries(model, input_tensors, output_tensors, index, count):
	"""
	Plots adversarial examples by applying the gradient of the loss with respect to the input.

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int

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
		index: int

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
		index: int

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
		index: int

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

def adversarial_test(dataloader, model, count=0):
	size = len(dataloader.dataset)	
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for x, y in dataloader:
			x, y = x.to(device), y.to(device)
			break

	inputs, gradxinputs = [], []
	for i in range(10):
		# generate_input(model, x, y, i, count=i)
		plot_adversaries(model, x, y, i, count=i)
	model.train()
	return

def test(dataloader, model, count=0, short=True):
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


epochs = 40
# generator = InvertedMediumNet()
# discriminator = MediumNetwork()
model = MediumNetworkFull()
loss_fn = nn.CrossEntropyLoss() 
# d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
# g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
# train(train_dataloader, model, loss_fn, optimizer, epochs)
# torch.save(model.state_dict(), 'trained_models/flowernet.pth')

model.load_state_dict(torch.load('trained_models/flowernet.pth'))
# test(test_dataloader, model)
adversarial_test(test_dataloader, model, 0)
# train_generative_adversaries(train_dataloader, discriminator, d_optimizer, generator, g_optimizer, loss_fn, epochs)


generator.load_state_dict(torch.load('trained_models/generator_large.pth'))
discriminator.load_state_dict(torch.load('trained_models/discriminator_large.pth'))


with torch.no_grad():
	for batch, (x, y) in enumerate(train_dataloader):
		_ = discriminator(x)
		x = x[0].reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
		plt.imshow(x)
		plt.show()
		plt.close()
		break
	random_output = torch.randn(16, 5)
	for i in range(1):
		random_output[0][0] += 0.01
		generated_outputs = generator(random_output)
		plt.imshow((generated_outputs[0] / torch.max(generated_outputs[0])).reshape(3, 256, 256).permute(1, 2, 0).detach().numpy())
		plt.tight_layout()
		plt.savefig('shifted_gan{0:04d}'.format(i), dpi=410)
		plt.close()



	for batch, (x, y) in enumerate(train_dataloader):
		_ = discriminator(x)
		x = x[0].reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
		plt.imshow(x)
		plt.show()
		plt.close()
		break

	random_output = torch.randn(16, 5)
	for i in range(1, 2):
		random_output[0][0] += 0.01
		generated_outputs = generator(random_output)
		plt.imshow((generated_outputs[0] / torch.max(generated_outputs[0])).reshape(3, 256, 256).permute(1, 2, 0).detach().numpy())
		plt.tight_layout()
		plt.savefig('shifted_gan{0:04d}'.format(i), dpi=410)
		plt.close()






