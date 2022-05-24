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
# from google.colab import files

# files.upload() # upload flower_photos_2
# !unzip flower_photos_2.zip

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

# specify batch size
minibatch_size = 64
train_data = ImageDataset(data_dir, image_type='.jpg')
dataloader = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class FCnet(nn.Module):

	def __init__(self):

		super().__init__()
		self.input_transform = nn.Linear(64*64*3, 64*64)
		self.d1 = nn.Linear(64*64, 2048)
		self.d2 = nn.Linear(2048, 512)
		self.d3 = nn.Linear(512, 1)
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
	plt.show()
	plt.savefig('gan_set{0:04d}.png'.format(count), dpi=410)
	plt.close()
	return

def train_generative_adversaries(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs):
	discriminator.train()
	generator.train()
	count = 0
	total_loss = 0
	start = time.time()
	fixed_input = torch.randn(minibatch_size, 2).to(device)
	fixed_input = torch.randn(minibatch_size, 100).to(device)

	for e in range(epochs):
		print (f"Epoch {e+1} \n" + '~'*100)
		print ('\n')
		for batch, (x, y) in enumerate(dataloader):
			if len(x) < minibatch_size:
				continue
			x = x.to(device)
	
			x = torch.flatten(x, start_dim=1)
			count += 1
			random_output = torch.randn(minibatch_size, 100).to(device)
			generated_samples = generator(random_output)
			input_dataset = torch.cat([x, generated_samples]) # or torch.cat([x, torch.rand(x.shape)])
			output_labels = torch.cat([torch.ones(len(y)), torch.zeros(len(generated_samples))]).to(device)
			discriminator_prediction = discriminator(input_dataset).reshape(minibatch_size*2)
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)

			discriminator_optimizer.zero_grad()
			discriminator_loss.backward()
			discriminator_optimizer.step()

			generated_outputs = generator(random_output)
			discriminator_outputs = discriminator(generated_outputs).reshape(minibatch_size)
			generator_loss = loss_fn(discriminator_outputs, torch.ones(len(y)).to(device)) # pretend that all generated inputs are in the dataset

			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()


		ave_loss = float(total_loss) / count
		elapsed_time = time.time() - start
		print (f"Average Loss: {ave_loss:.04}")
		print (f"Completed in {int(elapsed_time)} seconds")
		start = time.time()
		if e % 200 == 0:
			images = generator(fixed_input)
			print (images.shape)
			images = images.reshape(minibatch_size, 3, 64, 64).permute(0, 2, 3, 1).cpu().detach().numpy()
			show_batch(images, 0, grayscale=True)
			torch.save(discriminator.state_dict(), 'flower_discriminator.pth')
			torch.save(generator.state_dict(), 'flower_generator.pth')
	return

def input_gradient(model, input_tensor, output_dim):
	# change output to float
	input_tensor.requires_grad = True
	output = model.forward(input_tensor)

	# only scalars may be assigned a gradient
	output = torch.median(output)

	# backpropegate output gradient to input
	output.backward(retain_graph=True)
	
	return input_tensor.grad

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

epochs = 3000
discriminator = FCnet().to(device)
generator = InvertedFC().to(device)
count_parameters(discriminator)

loss_fn = nn.BCELoss()
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
train_generative_adversaries(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs)

torch.save(discriminator.state_dict(), 'discriminator.pth')
torch.save(generator.state_dict(), 'generator.pth')
# download checkpoint file
files.download('discriminator.pth')
files.download('generator.pth') 

discriminator.load_state_dict(torch.load('discriminator.pth'))
generator.load_state_dict(torch.load('generator.pth'))
fixed_input = torch.tensor([[0.,  0.]]).to(device)
print (fixed_input)
original_input = fixed_input.clone()

for i in range(16):
  for j in range(16):
    next_input = original_input.clone()
    next_input[0][0] = 1 - (1/8) * (i) + original_input[0][0]
    next_input[0][1] = -1 + (1/8) * (j) + original_input[0][1]
    fixed_input = torch.cat([fixed_input, next_input])

fixed_input = fixed_input[1:]

# input = fixed_input[-1] # deep in the 1s

while len(fixed_input) < 128:
  fixed_input = torch.cat([fixed_input, next_input])

# fixed_input = torch.randn(128, 2)
fixed_input = fixed_input.to(device)
fixed_outputs = generator(fixed_input)
inputs = fixed_outputs.reshape(16*16, 28, 28).cpu().detach().numpy()
show_batch(inputs, 0, grayscale=True)

torch.save(discriminator.state_dict(), 'discriminator.pth')
torch.save(generator.state_dict(), 'generator.pth')
# download checkpoint file
files.download('discriminator.pth')
files.download('generator.pth')

# fixed_input = torch.tensor([-0.4878,  2.4633]).to(device)
# new_output = generator(fixed_input)
# adversarial_input = new_output.reshape(28, 28).cpu().detach().numpy()
# plt.imshow(adversarial_input)
# plt.show()
# print (fixed_input)
# input_grad = input_gradient(generator, fixed_input, 784)
# adversarial_input = fixed_input
# adversarial_input = adversarial_input + 0.1 * torch.sign(input_grad)

# print (adversarial_input)
# new_output = generator(adversarial_input)
# adversarial_input = new_output.reshape(28, 28).cpu().detach().numpy()
# plt.imshow(adversarial_input)
# plt.show()
# plt.close()


