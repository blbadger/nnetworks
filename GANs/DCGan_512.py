# GAN_cnn_stabilized.py
# DCGAN -style convolutional generative adversarial network

# import standard libraries
import time
import pathlib
import os
import pandas as pd 
import random

# import third party libraries
import numpy as np 
from prettytable import PrettyTable
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt  
import torchvision.transforms as transforms

# dataset directory specification
data_dir = pathlib.Path('landscapes',  fname='Combined')

image_count = len(list(data_dir.glob('*.jpg')))

class ImageDataset(Dataset):
	"""
	Creates a dataset from images classified by folder name.  Random
	sampling of images to prevent overfitting
	"""

	def __init__(self, img_dir, transform=None, target_transform=None, image_type='.png'):
		# specify image labels by folder name 
		self.img_labels = [item.name for item in data_dir.glob('*')]

		# construct image name list: randomly sample images for each epoch
		images = list(img_dir.glob('*' + image_type))
		self.image_name_ls = images[:200]

		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.image_name_ls)

	def __getitem__(self, index):
		# path to image
		img_path = os.path.join(self.image_name_ls[index])
		image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB) # convert image to tensor of ints , torchvision.io.ImageReadMode.GRAY
		image = image / 255. # convert ints to floats in range [0, 1]
		image = torchvision.transforms.CenterCrop([512, 512])(image)

		# assign label to be a tensor based on the parent folder name
		label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image

# specify batch size
minibatch_size = 64
train_data = ImageDataset(data_dir, image_type='.jpg')
dataloader = DataLoader(train_data, batch_size=minibatch_size, shuffle=False)
print (len(dataloader))

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class StableDiscriminator(nn.Module):

	def __init__(self):
		super(StableDiscriminator, self).__init__()
		# switch second index to 3 for color
		self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1) # 3x128x128 image input
		self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(512, 1024, 4, stride=2, padding=1)
		self.conv6 = nn.Conv2d(1024, 1, 4, stride=1, padding=0)
		self.flatten = nn.Flatten(start_dim=1)
		self.fc = nn.Linear(13*13, 1)

		self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
		self.batchnorm2 = nn.BatchNorm2d(128)
		self.batchnorm3 = nn.BatchNorm2d(256)
		self.batchnorm4 = nn.BatchNorm2d(512)
		self.batchnorm5 = nn.BatchNorm2d(1024)
		self.sigmoid = nn.Sigmoid()

	def forward(self, input):
		out = self.conv1(input)
		out = self.leakyrelu(out)

		out = self.conv2(out)
		out = self.leakyrelu(out)
		out = self.batchnorm2(out)
		
		out = self.conv3(out)
		out = self.leakyrelu(out)
		out = self.batchnorm3(out)

		out = self.conv4(out)
		out = self.leakyrelu(out)
		out = self.batchnorm4(out)

		out = self.conv5(out)
		out = self.leakyrelu(out)
		out = self.batchnorm5(out)

		out = self.conv6(out)
		out = self.flatten(out)

		out = self.fc(out)
		out = self.sigmoid(out)
		return out


class StableGenerator(nn.Module):

	def __init__(self, minibatch_size):
		super(StableGenerator, self).__init__()
		self.input_transform = nn.ConvTranspose2d(1000, 2048, 4, 1, padding=0) # expects an input of shape 1x1000
		self.fc_transform = nn.Linear(100, 1024*4*4) # alternative as described in paper
		self.conv1 = nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1) 
		self.conv2 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
		self.conv3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
		self.conv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
		self.conv5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
		self.conv6 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
		# switch second index to 3 for color images
		self.conv7 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1) # end with shape minibatch_sizex3x256x256

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.minibatch_size = minibatch_size
		self.batchnorm1 = nn.BatchNorm2d(1024)
		self.batchnorm2 = nn.BatchNorm2d(512)
		self.batchnorm3 = nn.BatchNorm2d(256)
		self.batchnorm4 = nn.BatchNorm2d(128)
		self.batchnorm5 = nn.BatchNorm2d(64)

	def forward(self, input):
		input = input.reshape(minibatch_size, 1000, 1, 1)
		transformed_input = self.input_transform(input)
		# transformed_input = self.fc_transform(input).reshape(minibatch_size, 1024, 4, 4)
		out = self.conv1(transformed_input)
		out = self.relu(out)
		out = self.batchnorm1(out)

		out = self.conv2(out)
		out = self.relu(out)
		out = self.batchnorm2(out)

		out = self.conv3(out)
		out = self.relu(out)
		out = self.batchnorm3(out)

		out = self.conv4(out)
		out = self.relu(out)
		out = self.batchnorm4(out)

		out = self.conv5(out)
		out = self.relu(out)
		out = self.batchnorm5(out)

		out = self.conv6(out)
		out = self.relu(out)

		out = self.conv7(out)
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
	for n in range(8*8):
		ax = plt.subplot(8, 8, n+1)
		plt.axis('off')
		if grayscale:
			plt.imshow(input_batch[n], cmap='gray')
		else:
			plt.imshow(input_batch[n])
		plt.tight_layout()

	plt.style.use('dark_background')
	plt.tight_layout()
	# plt.show()
	plt.savefig('gan_set{0:04d}.png'.format(count), dpi=350)
	plt.close()
	return


def train_dcgan_adversaries(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs):
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
	fixed_input = torch.randn(minibatch_size, 1000).to(device)


	for e in range(epochs):
		print (f"Epoch {e+1} \n" + "~"*100)
		for batch, x in enumerate(dataloader):
			if len(x) < minibatch_size:
				break
			x = x.to(device)  
			count += 1
			# initialization
			discriminator_optimizer.zero_grad()
			random_output = torch.randn(minibatch_size, 1000).to(device)

			# train discriminator on real samples
			output_labels = torch.ones(len(x)).to(device)
			discriminator_prediction = discriminator(x).reshape(minibatch_size)
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)
			discriminator_loss.backward()

			# train discriminator on generated samples
			output_labels = torch.zeros(len(x)).to(device)
			generated_samples = generator(random_output)
			discriminator_prediction = discriminator(generated_samples).reshape(minibatch_size)
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)
			discriminator_loss.backward()
			discriminator_optimizer.step()

			generated_outputs = generator(random_output)
			discriminator_outputs = discriminator(generated_outputs).reshape(minibatch_size)
			generator_loss = loss_fn(discriminator_outputs, torch.ones(len(x)).to(device)) # pretend that all generated inputs are in the dataset

			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()
			print (generator_loss.item(), discriminator_loss.item())


		if discriminator_loss.item() > 0:
			torch.save(discriminator.state_dict(), 'flower_discriminator.pth')
			torch.save(generator.state_dict(), 'flower_generator.pth')

		fixed_outputs = generator(fixed_input)
		inputs = fixed_outputs.reshape(minibatch_size, 3, 512, 512).permute(0, 2, 3, 1).cpu().detach().numpy()
		show_batch(inputs, e, grayscale=True)
		# print (discriminator_prediction)
		print ('Generator loss: ', generator_loss)
		print ('Discriminator loss: ', discriminator_loss)

		ave_loss = float(total_loss) / count
		elapsed_time = time.time() - start
		print (f"Average Loss: {ave_loss:.04}")
		print (f"Completed in {int(elapsed_time)} seconds")
		start = time.time()
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

	images = generator(final_input[1:101].to(device)).reshape(len(fixed_input[0]), 3, 64, 64).permute(0, 2, 3, 1).cpu().detach().numpy()
	show_batch(images, 999, grayscale=True)
	return

class NewResNet(nn.Module):

	def __init__(self, model, num_classes):
		super().__init__()
		self.model = model
		self.fc = nn.Linear(512 * 4, num_classes)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		x = self.model.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)

		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)
		x = self.model.layer4(x)

		x = self.model.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.sigmoid(self.fc(x))
		return x


def count_parameters(model):
	table = PrettyTable(['Modules', 'Parameters'])
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


if __name__ == '__main__':
	epochs = 500
	discriminator = StableDiscriminator().to(device)
	# discriminator = NewResNet(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True), 1).to(device)
	generator = StableGenerator(minibatch_size).to(device)
	print ('Discriminator: ', count_parameters(discriminator))
	print ('Generator', count_parameters(generator))
	loss_fn = nn.BCELoss()
	discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
	generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

	train_dcgan_adversaries(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs)
	torch.save(discriminator.state_dict(), 'flower_discriminator.pth')
	torch.save(generator.state_dict(), 'flower_generator.pth')

	files.download('flower_discriminator.pth')
	files.download('flower_generator.pth')

	generator.load_state_dict(torch.load('trained_models/fmnist_fcgenerator.pth'))
	discriminator.load_state_dict(torch.load('trained_models/fmnist_fcdiscriminator.pth'))

	explore_latentspace(generator)

# discriminator = StableDiscriminator().to(device)
# generator = StableGenerator(minibatch_size).to(device)
# generator.load_state_dict(torch.load('flower_generator.pth'))
# discriminator.load_state_dict(torch.load('flower_discriminator.pth'))

# explore_latentspace(generator)

