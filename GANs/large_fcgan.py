# large_fcgan.py
# MLP-style model with GPU acceleration for latent space exploration.

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
from prettytable import PrettyTable
from torchvision import datasets

# files.upload() # upload flower_photos_2
# !unzip flower_photos_2.zip

# dataset directory specification
# data_dir = pathlib.Path('../flower_photos_2',  fname='Combined')

# image_count = len(list(data_dir.glob('*/*.jpg')))
# class_names = [item.name for item in data_dir.glob('*') 
# 			   if item.name not in ['._.DS_Store', '._DS_Store', '.DS_Store']]


transform = transforms.Compose(
	[transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.ToTensor()])

batch_size = 256 # global variable
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

def npy_loader(path):
	sample = torch.from_numpy(np.load(path))
	sample = sample.permute(0, 3, 2, 1)

	# 270* rotation
	for i in range(3):
		sample = torch.rot90(sample, dims=[2, 3])
	return sample / 255.

path = pathlib.Path('lsun_churches/churches/church_outdoor_train_lmdb_color_64.npy',  fname='Combined')
dataset = npy_loader(path)
dset = torch.utils.data.TensorDataset(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

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
		image = torchvision.transforms.Resize(size=[64, 64])(image)	
		image = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)	

		# assign label to be a tensor based on the parent folder name
		label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

		# convert image label to tensor
		label_tens = torch.tensor(self.img_labels.index(label))
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image, label_tens

# # specify batch size
# minibatch_size = 64
# train_data = ImageDataset(data_dir, image_type='.jpg')
# dataloader = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)

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
		out = self.sigmoid(out)
		return out


class StableGenerator(nn.Module):

	def __init__(self, minibatch_size):
		super(StableGenerator, self).__init__()
		self.input_transform = nn.ConvTranspose2d(100, 1024, 4, 1, padding=0) # expects an input of shape 1x100
		self.fc_transform = nn.Linear(100, 1024*4*4) # alternative as described in paper
		self.conv1 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1) 
		self.conv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
		self.conv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
		self.conv4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
		# switch second index to 3 for color images
		self.conv5 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1) # end with shape minibatch_sizex3x128x128

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.minibatch_size = batch_size
		self.batchnorm1 = nn.BatchNorm2d(512)
		self.batchnorm2 = nn.BatchNorm2d(256)
		self.batchnorm3 = nn.BatchNorm2d(128)
		self.batchnorm4 = nn.BatchNorm2d(64)


	def forward(self, input):
		input = input.reshape(batch_size, 100, 1, 1)
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
		out = self.tanh(out)
		return out


class FCnet(nn.Module):

	def __init__(self):

		super().__init__()
		self.input_transform = nn.Linear(64*64*3, 64*64)
		self.d1 = nn.Linear(64*64, 2048)
		self.d2 = nn.Linear(2048, 512)
		self.d3 = nn.Linear(512, 1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(0.)

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
		self.input_transform = nn.Linear(64*64, 64*64*3)
		self.d3 = nn.Linear(2048, 64*64)
		self.d2 = nn.Linear(512, 2048)
		self.d1 = nn.Linear(100, 512)
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


class DeepFC(nn.Module):

	def __init__(self, hidden_dim):

		super().__init__()
		self.input_transform = nn.Linear(64*64*3, hidden_dim)
		self.d1 = nn.Linear(hidden_dim, hidden_dim)
		self.d2 = nn.Linear(hidden_dim, hidden_dim)
		self.d3 = nn.Linear(hidden_dim, hidden_dim)
		self.d4 = nn.Linear(hidden_dim, hidden_dim)
		self.d5 = nn.Linear(hidden_dim, hidden_dim)
		self.d6 = nn.Linear(hidden_dim, hidden_dim)
		self.d7 = nn.Linear(hidden_dim, hidden_dim)
		self.d8 = nn.Linear(hidden_dim, hidden_dim)
		self.d9 = nn.Linear(hidden_dim, hidden_dim)
		self.d10 = nn.Linear(hidden_dim, 1)
		self.relu = nn.ReLU()
		self.gelu = nn.GELU()
		self.layernorm1 = nn.LayerNorm(hidden_dim)
		self.layernorm2 = nn.LayerNorm(hidden_dim)
		self.layernorm3 = nn.LayerNorm(hidden_dim)
		self.layernorm4 = nn.LayerNorm(hidden_dim)
		self.layernorm5 = nn.LayerNorm(hidden_dim)
		self.layernorm6 = nn.LayerNorm(hidden_dim)
		self.layernorm7 = nn.LayerNorm(hidden_dim)
		self.layernorm8 = nn.LayerNorm(hidden_dim)
		self.layernorm9 = nn.LayerNorm(hidden_dim)
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(0.)

	def forward(self, input_tensor):
		out = self.input_transform(input_tensor)
		out = self.gelu(out)

		out = self.d1(out)
		out = self.gelu(out)
		out = self.layernorm1(out)

		out = self.d2(out)
		out = self.gelu(out)
		out = self.layernorm2(out)

		out = self.d3(out)
		out = self.gelu(out)
		out = self.layernorm3(out)

		out = self.d4(out)
		out = self.gelu(out)
		out = self.layernorm4(out)

		out = self.d5(out)
		out = self.gelu(out)
		out = self.layernorm5(out)

		out = self.d6(out)
		out = self.gelu(out)
		out = self.layernorm6(out)

		out = self.d7(out)
		out = self.gelu(out)
		out = self.layernorm7(out)

		out = self.d8(out)
		out = self.gelu(out)
		out = self.layernorm8(out)

		out = self.d9(out)
		out = self.gelu(out)
		out = self.layernorm9(out)

		out = self.d10(out)
		out = self.sigmoid(out)

		return out

class DeepGen(nn.Module):

	def __init__(self, hidden_dim):

		super().__init__()
		self.input_transform = nn.Linear(hidden_dim, 64*64*3)
		self.d1 = nn.Linear(hidden_dim, hidden_dim)
		self.d2 = nn.Linear(hidden_dim, hidden_dim)
		self.d3 = nn.Linear(hidden_dim, hidden_dim)
		self.d4 = nn.Linear(hidden_dim, hidden_dim)
		self.d5 = nn.Linear(hidden_dim, hidden_dim)
		self.d6 = nn.Linear(hidden_dim, hidden_dim)
		self.d7 = nn.Linear(hidden_dim, hidden_dim)
		self.d8 = nn.Linear(hidden_dim, hidden_dim)
		self.d9 = nn.Linear(hidden_dim, hidden_dim)
		self.d10 = nn.Linear(hidden_dim, 100)
		self.relu = nn.ReLU()
		self.gelu = nn.GELU()
		self.layernorm1 = nn.LayerNorm(hidden_dim)
		self.layernorm2 = nn.LayerNorm(hidden_dim)
		self.layernorm3 = nn.LayerNorm(hidden_dim)
		self.layernorm4 = nn.LayerNorm(hidden_dim)
		self.layernorm5 = nn.LayerNorm(hidden_dim)
		self.layernorm6 = nn.LayerNorm(hidden_dim)
		self.layernorm7 = nn.LayerNorm(hidden_dim)
		self.layernorm8 = nn.LayerNorm(hidden_dim)
		self.layernorm9 = nn.LayerNorm(hidden_dim)
		self.tanh = nn.Tanh()
		self.dropout = nn.Dropout(0.)

	def forward(self, input_tensor):
		out = self.input_transform(input_tensor)
		out = self.gelu(out)

		out = self.d1(out)
		out = self.gelu(out)
		out = self.layernorm1(out)

		out = self.d2(out)
		out = self.gelu(out)
		out = self.layernorm2(out)

		out = self.d3(out)
		out = self.gelu(out)
		out = self.layernorm3(out)

		out = self.d4(out)
		out = self.gelu(out)
		out = self.layernorm4(out)

		out = self.d5(out)
		out = self.gelu(out)
		out = self.layernorm5(out)

		out = self.d6(out)
		out = self.gelu(out)
		out = self.layernorm6(out)

		out = self.d7(out)
		out = self.gelu(out)
		out = self.layernorm7(out)

		out = self.d8(out)
		out = self.gelu(out)
		out = self.layernorm8(out)

		out = self.d9(out)
		out = self.gelu(out)
		out = self.layernorm9(out)

		out = self.d10(out)
		out = self.sigmoid(out)

		out = self.input_transform(out)

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
			plt.imshow(input_batch[n])
		else:
			plt.imshow(input_batch[n])
		plt.tight_layout()

	plt.tight_layout()
	# plt.show()
	plt.savefig('gan_set{0:04d}.png'.format(count), dpi=410)
	plt.close()
	return

def train_generative_adversaries(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs):
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
	fixed_input = torch.randn(batch_size, 2).to(device)
	fixed_input = torch.randn(batch_size, 100).to(device)

	for e in range(epochs):
		print (f"Epoch {e+1} \n" + '~'*100)
		for batch, x in enumerate(dataloader):
			if len(x) < batch_size:
				continue
			x = x.to(device)
			x = torch.flatten(x, start_dim=1)
			count += 1
			random_output = torch.randn(batch_size, 100).to(device)
			generated_samples = generator(random_output)
			input_dataset = torch.cat([x, generated_samples]) # concatenated sample approach
			output_labels = torch.cat([torch.ones(len(x)), torch.zeros(len(generated_samples))]).to(device)
			discriminator_prediction = discriminator(input_dataset).reshape(batch_size*2)
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)

			discriminator_optimizer.zero_grad()
			discriminator_loss.backward()
			discriminator_optimizer.step()

			generated_outputs = generator(random_output)
			discriminator_outputs = discriminator(generated_outputs).reshape(batch_size)
			generator_loss = loss_fn(discriminator_outputs, torch.ones(len(x)).to(device)) # pretend that all generated inputs are in the dataset
			print (generator_loss.item(), discriminator_loss.item())

			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()


		ave_loss = float(total_loss) / count
		elapsed_time = time.time() - start
		print (f"Average Loss: {ave_loss:.04}")
		print (f"Completed in {int(elapsed_time)} seconds")
		start = time.time()
		images = generator(fixed_input)
		images = images.reshape(batch_size, 3, 64, 64).permute(0, 2, 3, 1).cpu().detach().numpy()
		show_batch(images, e, grayscale=False)
		torch.save(discriminator.state_dict(), 'flower_discriminator.pth')
		torch.save(generator.state_dict(), 'flower_generator.pth')
	return

def input_gradient(model, input_tensor):
	"""
	Finds the gradient of the output's max val w.r.t the input.

	Args:
		model: torch.nn.Module() object
		input_tensor: torch.tensor
	Returns:
		gradient: torch.tensor.grad object
	"""

	# change output to float
	input_tensor.requires_grad = True
	output = model.forward(input_tensor)

	# only scalars may be assigned a gradient
	output = torch.max(output)

	# backpropegate output gradient to input
	output.backward(retain_graph=True)
	gradient = input_tensor.grad
	return input_tensor

def count_parameters(model):
	"""
	Display the tunable parameters in the model of interest

	Args:
		model: torch.nn object

	Returns:
		total_params: the number of model parameters

	"""

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

def latent_space_2d():
	"""
	Two-dimensional exploration of latent space. Assumes 2-dimensional space

	Args:
		None

	Returns:
		None
	"""

	fixed_input = torch.tensor([[0.,  0.]]).to(device)
	original_input = fixed_input.clone()

	for i in range(16):
	  for j in range(16):
	    next_input = original_input.clone()
	    next_input[0][0] = 1 - (1/8) * (i) + original_input[0][0]
	    next_input[0][1] = -1 + (1/8) * (j) + original_input[0][1]
	    fixed_input = torch.cat([fixed_input, next_input])

	fixed_input = fixed_input[1:]
	while len(fixed_input) < 128:
	  fixed_input = torch.cat([fixed_input, next_input])

	# fixed_input = torch.randn(128, 2)
	fixed_input = fixed_input.to(device)
	fixed_outputs = generator(fixed_input)
	inputs = fixed_outputs.reshape(16*16, 28, 28).cpu().detach().numpy()
	show_batch(inputs, 0, grayscale=True)

	return

def latent_space():
	"""
	Plot a 10x10 grid of outputs from the latent space (assumes 100-dimensional space)

	Args:
		None

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
	epochs = 300
	discriminator = FCnet().to(device)
	generator = InvertedFC().to(device)

	# discriminator = StableDiscriminator().to(device)
	# generator = StableGenerator(batch_size).to(device)
	count_parameters(discriminator)

	loss_fn = nn.BCELoss()
	discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
	generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
	train_generative_adversaries(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs)

	torch.save(discriminator.state_dict(), 'discriminator.pth')
	torch.save(generator.state_dict(), 'generator.pth')

	# # download checkpoint file (for use in colab)
	# files.download('discriminator.pth')
	# files.download('generator.pth') 
	# latent_space()
	


