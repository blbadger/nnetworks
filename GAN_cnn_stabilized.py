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
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt  
import torchvision.transforms as transforms

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
		image = torchvision.transforms.Resize(size=[128, 128])(image)	
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
minibatch_size = 128
train_data = ImageDataset(data_dir, image_type='.jpg')
train_dataloader = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)

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
		self.minibatch_size = minibatch_size
		self.batchnorm1 = nn.BatchNorm2d(512)
		self.batchnorm2 = nn.BatchNorm2d(256)
		self.batchnorm3 = nn.BatchNorm2d(128)
		self.batchnorm4 = nn.BatchNorm2d(64)


	def forward(self, input):
		input = input.reshape(minibatch_size, 100, 1, 1)
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
	plt.show()
	plt.savefig('gan_set{0:04d}.png'.format(count), dpi=410)
	plt.close()
	return


def train_dcgan_adversaries(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs):
	discriminator.train()
	generator.train()
	count = 0
	total_loss = 0
	start = time.time()
	fixed_input = torch.randn(minibatch_size, 100).to(device)

	for e in range(epochs):
		print (f"Epoch {e+1} \n" + "~"*100)
		for batch, (x, y) in enumerate(dataloader):
			x = x.to(device)
			show_batch(x, count)
			if len(x) < minibatch_size:
				break
			count += 1
			print (count)
			# initialization
			discriminator_optimizer.zero_grad()
			random_output = torch.randn(minibatch_size, 100).to(device)

			# train discriminator on real samples
			output_labels = torch.ones(len(y)).to(device)
			discriminator_prediction = discriminator(x).reshape(minibatch_size)
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)
			discriminator_loss.backward()

			# train discriminator on generated samples
			output_labels = torch.zeros(len(y)).to(device)
			generated_samples = generator(random_output)
			discriminator_prediction = discriminator(generated_samples).reshape(minibatch_size)
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)
			discriminator_loss.backward()
			discriminator_optimizer.step()

			generated_outputs = generator(random_output)
			discriminator_outputs = discriminator(generated_outputs).reshape(minibatch_size)
			generator_loss = loss_fn(discriminator_outputs, torch.ones(len(y)).to(device)) # pretend that all generated inputs are in the dataset

			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()

		if e % 200 == 0:
			images = generator(fixed_input)
			print (images.shape)
			images = images.reshape(128, 3, 128, 128).permute(0, 2, 3, 1).cpu().detach().numpy()
			show_batch(images, 0, grayscale=True)
			torch.save(discriminator.state_dict(), 'flower_discriminator.pth')
			torch.save(generator.state_dict(), 'flower_generator.pth')

		fixed_outputs = generator(fixed_input)
		inputs = fixed_outputs.reshape(minibatch_size, 3, 128, 128).permute(0, 2, 3, 1).cpu().detach().numpy()

		# show_batch(inputs, e, grayscale=True)
		# print (discriminator_prediction)
		print ('Generator loss: ', generator_loss)
		print ('Discriminator loss: ', discriminator_loss)

		ave_loss = float(total_loss) / count
		elapsed_time = time.time() - start
		print (f"Average Loss: {ave_loss:.04}")
		print (f"Completed in {int(elapsed_time)} seconds")
		start = time.time()
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

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

epochs = 3000
discriminator = StableDiscriminator().to(device)
generator = StableGenerator(minibatch_size).to(device)
# discriminator.apply(weights_init)
# generator.apply(weights_init)
loss_fn = nn.BCELoss()
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
# train(train_dataloader, trained_model, loss_fn, generator_optimizer, epochs)
# discriminator.load_state_dict(torch.load('trained_models/discriminator.pth'))
# generator.load_state_dict(torch.load('trained_models/generator.pth'))

train_dcgan_adversaries(train_dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer, loss_fn, epochs)
torch.save(discriminator.state_dict(), 'flower_discriminator.pth')
torch.save(generator.state_dict(), 'flower_generator.pth')

files.download('flower_discriminator.pth')
files.download('flower_generator.pth')

# generator.load_state_dict(torch.load('trained_models/fmnist_fcgenerator.pth'))
# discriminator.load_state_dict(torch.load('trained_models/fmnist_fcdiscriminator.pth'))

fixed_input = torch.randn(1, 100, 1, 1)

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

fixed_input = torch.randn(128, 100, 1, 1).to(device)
# images = generator(final_input[1:101])
images = generator(fixed_input)
print (images.shape)
images = images.reshape(128, 3, 128, 128).permute(0, 2, 3, 1).cpu().detach().numpy()
show_batch(images, 0, grayscale=True)
