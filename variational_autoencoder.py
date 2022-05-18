# variational_autoencoder.py

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


# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

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
		image = torchvision.io.read_image(img_path) # convert image to tensor of ints , torchvision.io.ImageReadMode.GRAY
		image = image / 255. # convert ints to floats in range [0, 1]
		image = torchvision.transforms.Resize(size=[28, 28])(image)	

		# assign label to be a tensor based on the parent folder name
		label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

		# convert image label to tensor
		label_tens = torch.tensor(self.img_labels.index(label))
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image, label_tens


tensorify_and_normalize = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.1)]
)

# specify batch size
minibatch_size = 128
train_data = torchvision.datasets.MNIST(
	root = '.',
	train = True,
	transform = tensorify_and_normalize,
  download = True
	)

dataloader = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)

class Autoencoder(nn.Module):

	def __init__(self):
		super().__init__()
		self.d1= nn.Linear(28*28, 1024)
		self.d2 = nn.Linear(1024, 64)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.3)

	def forward(self, input_tensor):
		out = self.d1(input_tensor)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.d2(out)
		out = self.relu(out)
		return out

class Autodecoder(nn.Module):

	def __init__(self):
		super().__init__()
		self.d1 = nn.Linear(64, 1024)
		self.d2 = nn.Linear(1024, 28*28)
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()

		self.dropout = nn.Dropout(0.1)


	def forward(self, input_tensor):
		out = self.d1(input_tensor)
		out = self.relu(out)

		out = self.d2(out)
		out = self.sigmoid(out)
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

	plt.tight_layout()
	# plt.show()
	plt.savefig('vae_set{0:04d}.png'.format(count), dpi=410)
	plt.close()
	return

def train_autoencoder(encoder, encoder_optimizer, decoder, decoder_optimizer, loss_fn, epochs):
	encoder.train()
	decoder.train()
	start = time.time()

	for e in range(epochs):
		print (f"Epoch {e+1}")
		for batch, (x, y) in enumerate(dataloader):
			x, y = x.to(device), y.to(device)
			if len(x) < minibatch_size:
				continue
			x = torch.flatten(x, start_dim=1)
			latent_space = encoder(x)

			output = decoder(latent_space)
			decoder_optimizer.zero_grad()
			encoder_optimizer.zero_grad()
			loss = loss_fn(output, x)
			loss.backward()
			decoder_optimizer.step()
			encoder_optimizer.step()
	return

def test_decoder(decoder):
	decoder.eval()
	random_input = torch.randn(64, 64)
	output_batch = decoder(random_input)
	output_batch = output_batch.reshape(64, 28, 28).cpu().detach().numpy()
	show_batch(output_batch, grayscale=True)
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

epochs = 2
encoder = Autoencoder()
decoder = Autodecoder()
loss_fn = nn.MSELoss()
encoder_optimizer = torch.optim.Adam(encoder.parameters())
decoder_optimizer = torch.optim.Adam(decoder.parameters())
model = Autoencoder()
train_autoencoder(encoder, encoder_optimizer, decoder, decoder_optimizer, loss_fn, epochs)
test_decoder(decoder)


