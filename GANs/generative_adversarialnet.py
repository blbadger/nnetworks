# generative_adversarialnets.py
# GANs for image generation and latent space exploration

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
import torchvision.transforms as transforms


# dataset directory specification
data_dir = pathlib.Path('../flower_photos_2',  fname='Combined')
image_count = len(list(data_dir.glob('*/*.jpg')))
class_names = [item.name for item in data_dir.glob('*') 
			   if item.name not in ['._.DS_Store', '._DS_Store', '.DS_Store']]
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
minibatch_size = 64

def load_fmnist():
	train_data = torchvision.datasets.FashionMNIST(
		root = '../fashion_mnist/FashionMNIST',
		train = True,
		transform = torchvision.transforms.ToTensor()
		)

	test_data = torchvision.datasets.FashionMNIST(
		root = '../fashion_mnist/FashionMNIST',
		train = False,
		transform = torchvision.transforms.ToTensor()
		)

	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
				   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	return train_data, test_data, class_names


train_data = ImageDataset(data_dir, image_type='.jpg')
train_dataloader = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=minibatch_size, shuffle=False)

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")


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
		self.d1 = nn.Linear(2048, 512)
		self.d2 = nn.Linear(512, 50)
		self.d3 = nn.Linear(50, 1)
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(0.1)
		self.index1, self.index2, self.index3, self.index4 = [], [], [], []
		

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
		output = self.dropout(output)

		output = self.d2(output)
		output = self.relu(output)
		output = self.dropout(output)

		final_output = self.d3(output)
		final_output = self.sigmoid(final_output)
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
		self.tanh = nn.Tanh()
		self.d1 = nn.Linear(512, 2048)
		self.d2 = nn.Linear(50, 512)

	def forward(self, final_output):
		output = self.d2(final_output)
		output = self.relu(output)
		output = self.d1(output)
		output = self.relu(output)

		out = torch.reshape(output, (16, 32, 8, 8)) # reshape for convolutions
		out = self.max_pooling(out, discriminator.index4)
		out = self.relu(self.conv32(out))
		out = self.max_pooling(out, discriminator.index3)
		out = self.relu(self.conv16(out))
		out = self.max_pooling(out, discriminator.index2)
		out = self.relu(self.conv16(out))
		out = self.max_pooling(out, discriminator.index1)
		out = self.tanh(self.entry_conv(out))
		return out


class FCnet(nn.Module):

	def __init__(self):

		super().__init__()
		self.input_transform = nn.Linear(28*28*3, 1024)
		self.d1 = nn.Linear(1024, 512)
		self.d2 = nn.Linear(512, 256)
		self.d3 = nn.Linear(256, 1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(0.3)

	def forward(self, input_tensor):
		out = self.input_transform(input_tensor)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.d1(out)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.d2(out)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.d3(out)
		out = self.sigmoid(out)
		return out


class InvertedFC(nn.Module):

	def __init__(self):
		super().__init__()
		self.input_transform = nn.Linear(1024, 28*28*3)
		self.d3 = nn.Linear(512, 1024)
		self.d2 = nn.Linear(256, 512)
		self.d1 = nn.Linear(100, 256)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.tanh= nn.Tanh()

	def forward(self, input_tensor):
		out = self.d1(input_tensor)
		out = self.relu(out)

		out = self.d2(out)
		out = self.relu(out)

		out = self.d3(out)
		out = self.relu(out)

		out = self.input_transform(out)
		out = self.tanh(out)
		return out


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

	plt.style.use('dark_background')
	plt.tight_layout()
	plt.savefig('gan_set{0:04d}.png'.format(count), dpi=410)
	plt.close()
	return

def train_fcgan(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs):
	"""
	Trains the generative adversarial network model.

	Args:
		dataloader: torch.utils.data.Dataloader object, iterable for loading training and test data
		discriminator: torch.nn.Module() object
		discriminator_optimizer: torch.optim object, optimizes discriminator params during gradient descent
		generator: torch.nn.Module() object
		generator_optimizer: torc.optim object, optimizes generator params during gradient descent
		loss_fn: arbitrary method to apply to models (default binary cross-entropy loss)
		epochs: int, number of training epochs desired

	Returns:
		None (modifies generator and discriminator in-place)
	"""
	discriminator.train()
	generator.train()
	count = 0
	total_loss = 0
	start = time.time()
	fixed_input = torch.randn(minibatch_size, 100)

	for e in range(epochs):
		print (f"Epoch {e+1} \n" + '~'*100)
		for batch, (x, y) in enumerate(dataloader):
			if len(x) < minibatch_size:
				break

			x = torch.flatten(x, start_dim=1)
			count += 1
			random_output = torch.randn(minibatch_size, 100)
			generated_samples = generator(random_output)
			input_dataset = torch.cat([x, generated_samples]) # or torch.cat([x, torch.rand(x.shape)])
			output_labels = torch.cat([torch.ones(len(y)), torch.zeros(len(generated_samples))])
			discriminator_prediction = discriminator(input_dataset).reshape(minibatch_size*2)
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)

			discriminator_optimizer.zero_grad()
			discriminator_loss.backward()
			discriminator_optimizer.step()

			generated_outputs = generator(random_output)
			discriminator_outputs = discriminator(generated_outputs).reshape(minibatch_size)
			generator_loss = loss_fn(discriminator_outputs, torch.ones(len(y))) # pretend that all generated inputs are in the dataset

			regan = True
			if regan:
				discriminator_loss = -generator_loss
				discriminator_optimizer.zero_grad()
				discriminator_optimizer.backward()
				discriminator_optimizer.step()

			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()

			if count % 50 == 0:
				fixed_outputs = generator(fixed_input)
				inputs = fixed_outputs.reshape(minibatch_size, 3, 28, 28).permute(0, 2, 3, 1).detach().numpy()
				show_batch(inputs, count // 50,  grayscale=False)

		ave_loss = float(total_loss) / count
		elapsed_time = time.time() - start
		print (f"Average Loss: {ave_loss:.04}")
		print (f"Completed in {int(elapsed_time)} seconds")
	return


def train_convgan(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs):
	"""
	Trains the generative adversarial network model.

	Args:
		dataloader: torch.utils.data.Dataloader object, iterable for loading training and test data
		discriminator: torch.nn.Module() object
		discriminator_optimizer: torch.optim object, optimizes discriminator params during gradient descent
		generator: torch.nn.Module() object
		generator_optimizer: torc.optim object, optimizes generator params during gradient descent
		loss_fn: arbitrary method to apply to models (default binary cross-entropy loss)
		epochs: int, number of training epochs desired

	Returns:
		None (modifies generator and discriminator in-place)
	"""
	discriminator.train()
	generator.train()
	count = 0
	total_loss = 0
	start = time.time()
	fixed_input = torch.randn(minibatch_size, 50)

	for e in range(epochs):
		print (f"Epoch {e+1} \n" + "~"*100)
		print ('\n')
		for batch, (x, y) in enumerate(dataloader):
			print (x.shape)
			show_batch(x.reshape(minibatch_size, 3, 256, 256).permute(0, 2, 3, 1).detach().numpy(), e)
			count += 1
			_ = discriminator(x) # initialize the index arrays

			random_output = torch.randn(minibatch_size, 50)
			generated_samples = generator(random_output)
			input_dataset = torch.cat([x, generated_samples]) # or torch.cat([x, torch.rand(x.shape)])
			output_labels = torch.cat([torch.ones(len(y)), torch.zeros(len(generated_samples))])
			discriminator_prediction = discriminator(input_dataset).reshape(minibatch_size*2)
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)

			discriminator_optimizer.zero_grad()
			discriminator_loss.backward()
			discriminator_optimizer.step()

			_ = discriminator(x) # reset index dims to 16-element minibatch size
			generated_outputs = generator(random_output)
			discriminator_outputs = discriminator(generated_outputs).reshape(minibatch_size)
			generator_loss = loss_fn(discriminator_outputs, torch.ones(len(y))) # pretend that all generated inputs are in the dataset

			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()

			if count % 20 == 0:
				fixed_outputs = generator(fixed_input)
				inputs = (fixed_outputs / torch.max(fixed_outputs)).reshape(16, 3, 256, 256).permute(0, 2, 3, 1).detach().numpy()
				show_batch(inputs, count//20)
				print (generator.d2.bias.norm())
				print (discriminator_prediction)
				print (generator_loss)
				print (discriminator_loss)

		ave_loss = float(total_loss) / count
		elapsed_time = time.time() - start
		print (f"Average Loss: {ave_loss:.04}")
		print (f"Completed in {int(elapsed_time)} seconds")
		start = time.time()
		torch.save(generator.state_dict(), 'trained_models/generator.pth')
		torch.save(discriminator.state_dict(), 'trained_models/discriminator.pth')

	return

def generate_input(model, input_tensors, output_tensors, index, count):
	"""
	Generates an input for a desired output class

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
		input_grad = loss_gradient(model, single_input, output_tensors[index], 5) # compute input gradient
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


def explore_latentspace(generator):
	"""
	Plot a 10x10 grid of outputs from the latent space (assumes a 100x100 grid)

	Args:
		generator: torch.nn.Module() object of the generator

	Returns:
		None (saves png image in call to show_batch())
	"""
	fixed_input = torch.randn(1, 100)
	final_input = fixed_input.clone()
	for i in range(10):
		for j in range(10):
			new_input = fixed_input.clone()
			new_input[0][1:20] += 0.25 * (i+1)
			new_input[0][20:40] -= 0.25* (j+1)
			new_input[0][40:60] += 0.25 * (i+1)
			new_input[0][60:80] -= 0.25 * (j+1)
			new_input[0][80:100] += 0.25 * (j+1)
			final_input = torch.cat([final_input, new_input])

	images = generator(final_input[1:101]).reshape(len(fixed_input[0]), 28, 28).detach().numpy()
	show_batch(images, 0, grayscale=True)
	return

if __name__ == '__main__':
	epochs = 1000
	discriminator = FCnet()
	generator = InvertedFC()
	loss_fn = nn.BCELoss()
	discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
	generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

	train_colorgan_adversaries(train_dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs)
	torch.save(discriminator.state_dict(), 'trained_models/flower_discriminator.pth')
	torch.save(generator.state_dict(), 'trained_models/flower_generator.pth')

	generator.load_state_dict(torch.load('trained_models/fmnist_fcgenerator.pth'))
	discriminator.load_state_dict(torch.load('trained_models/fmnist_fcdiscriminator.pth'))

	explore_latentspace()
