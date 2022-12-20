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
import matplotlib.pyplot as plt  
import torchvision


class ImageDataset(Dataset):
	"""
	Creates a dataset from images classified by folder name.  Random
	sampling of images to prevent overfitting
	"""

	def __init__(self, img_dir, transform=None, target_transform=None):
		# specify image labels by folder name 
		self.img_labels = [item.name for item in data_dir.glob('*')]

		# construct image name list: randomly sample 400 images for each epoch
		images = list(img_dir.glob('*/*.png'))
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
		image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.GRAY) # convert image to tensor of ints as read in grayscale in range [0, 255]
		image = image / 255. # convert ints to floats in range [0, 1]
		image = torchvision.transforms.Resize(size=28)(image)	

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

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class DeepNetwork(nn.Module):

	def __init__(self):

		super(DeepNetwork, self).__init__()
		self.entry_conv = Conv2d(1, 16, 3, padding=(1, 1))
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
		self.d3 = nn.Linear(50, 10)
		

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
		self.entry_conv = Conv2d(1, 16, 3, padding=(1, 1))
		self.conv16 = Conv2d(16, 16, 3, padding=(1, 1))
		self.conv32 = Conv2d(16, 32, 3, padding=(1, 1))
		self.conv32_2 = Conv2d(32, 32, 3, padding=(1, 1))
		self.conv64 = Conv2d(32, 64, 3, padding=(1, 1))
		self.conv64_2 = Conv2d(64, 64, 3, padding=(1, 1))

		self.max_pooling = nn.MaxPool2d(2)
		self.flatten = nn.Flatten()
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)
		self.d1 = nn.Linear(32, 512)
		self.d2 = nn.Linear(512, 50)
		self.d3 = nn.Linear(50, 10)
		

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
		final_output = self.softmax(final_output)
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

	# backpropegate output gradient to input
	loss.backward(retain_graph=True)
	gradientxinput = torch.abs(input_tensor.grad) * input_tensor

	if max_normalized:
		max_val = torch.max(gradientxinput)
		if max_val > 0:
			gradientxinput = gradientxinput / max_val

	return gradientxinput

def train(dataloader, model, loss_fn, optimizer):
	model.train()
	count = 0
	total_loss = 0
	start = time.time()

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

def show_batch(input_batch, output_batch, gradxinput_batch, individuals=False, count=0):
	"""
	Show a batch of images with gradientxinputs superimposed

	Args:
		output_batch: arr[torch.Tensor] of input images

	"""
	if individuals:
		for n in range(len(input_batch)):
			ax = plt.subplot(1, 3, 1)
			plt.axis('off')
			plt.title('Input')
			plt.imshow(input_batch[n], cmap='gray', alpha=1)
			ax = plt.subplot(1, 3, 2)
			plt.axis('off')
			plt.title('Gradient * Input')
			plt.imshow(gradxinput_batch[n], cmap='inferno', alpha=1)
			ax = plt.subplot(1, 3, 3)
			plt.axis('off')
			plt.title('Combined')
			plt.imshow(input_batch[n], cmap='gray', alpha=1)
			plt.imshow(gradxinput_batch[n], cmap='inferno', alpha=0.5)
			plt.tight_layout()
			plt.savefig(f'attribution{n}.png', dpi=410)
			plt.close()

	plt.figure(figsize=(10, 10))
	for n in range(len(input_batch)):
		ax = plt.subplot(4, 4, n+1) # expects a batch of size 16
		plt.axis('off')
		plt.title(class_names[int(output_batch[n])].title())
		plt.imshow(input_batch[n], cmap='gray', alpha=1)
		plt.imshow(gradxinput_batch[n], cmap='inferno', alpha=0.5)

	plt.tight_layout()
	plt.savefig('attribution_grid{0:04d}.png'.format(count), dpi=410)
	return

def generate_adversaries(model, input_tensors, output_tensors, index):
	"""
	Plots adversarial examples by applying the gradient of the loss with respect to the input.

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int

	returns:
		None (saves .png image)
	"""

	single_input = input_tensors[index].reshape(1, 1, 28, 28)
	# input_grad = torch.sign(torch.rand(single_input.shape) - 0.5)
	# added_input = single_input + 0.1*input_grad

	added_input = torch.clone(single_input)
	for i in range(200):
		input_grad = torch.sign(loss_gradient(model, added_input, output_tensors[index], 10))
		# max_grad = torch.max(input_grad)
		# if max_grad == 0:
		# 	continue
		# input_grad = input_grad / max_grad
		added_input = added_input + 0.001*input_grad
		added_input = added_input.detach()

	original_pred = model(single_input)
	adversarial_pred = model(added_input)
	grad_pred = model(0.1*input_grad)
	input_img = single_input.reshape(28, 28).detach().numpy()
	gradient = 0.5*input_grad.reshape(28, 28).detach().numpy()
	added_input = added_input.reshape(28, 28).detach().numpy()
	actual_class = class_names[int(output_tensors[index])].title()

	original_class = class_names[int(original_pred.argmax(1))].title()
	original_confidence = int(max(original_pred.detach().numpy()[0]) * 100)

	adversarial_class = class_names[int(adversarial_pred.argmax(1))].title()
	adversarial_confidence = int(max(adversarial_pred.detach().numpy()[0]) * 100)

	grad_class = class_names[int(grad_pred.argmax(1))].title() 
	grad_confidence = int(max(grad_pred.detach().numpy()[0]) * 100)

	ax = plt.subplot(1, 3, 1)
	plt.axis('off')
	plt.title('{}% {}, {}'.format(original_confidence, original_class, actual_class))
	plt.imshow(input_img, alpha=1, cmap='gray')
	ax = plt.subplot(1, 3, 2)
	plt.axis('off')
	plt.title('Gradient Tensor')
	plt.imshow(gradient, alpha=1, cmap='gray')
	ax = plt.subplot(1, 3, 3)
	plt.axis('off')
	plt.title('{}% {}'.format(adversarial_confidence, adversarial_class))
	plt.imshow(added_input, alpha=1, cmap='gray')
	plt.tight_layout()
	plt.savefig(f'adversarial_example{index}.png', dpi=410)
	plt.close()

	if original_class != adversarial_class:
		return 1
	else:
		return 0

	return

def adversarial_test(dataloader, model, count=0):
	size = len(dataloader.dataset)	
	model.eval()
	test_loss, correct = 0, 0
	adversaries, total = 0, 0

	for x, y in dataloader:
		x, y = x.to(device), y.to(device)
		pred = model(x)
		correct += (pred.argmax(1) == y).type(torch.float).sum().item()

		inputs, gradxinputs = [], []
		for i in range(len(x)):
			adversaries += generate_adversaries(model, x, y, i)
			total += 1

	print (adversaries, total)
	adversarials = round(adversaries / total, 2) * 100
	print (f"Adversarial percentage: {adversarials}")
	model.train()
	return


def test(dataloader, model, count=0):
	size = len(dataloader.dataset)	
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for x, y in dataloader:
			x, y = x.to(device), y.to(device)
			pred = model(x)
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
			break

	inputs, gradxinputs = [], []
	for i in range(len(x)):
		single_input= x[i].reshape(1, 1, 28, 28)
		gxi = gradientxinput(model, single_input, 10)
		input_img = single_input.reshape(28, 28).detach().numpy()
		gxi_img = gxi.reshape(28, 28).detach().numpy()
		inputs.append(input_img)
		gradxinputs.append(gxi_img)

	show_batch(inputs, y, gradxinputs, count=count)
	accuracy = correct / size
	print (f"Test accuracy: {int(correct)} / {size}")
	model.train()
	return


epochs = 3
model = MediumNetwork()
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters())
model.load_state_dict(torch.load('trained_models/fmnist_bench_pytorch.pth'))
# for e in range(epochs):
# 	print (f"Epoch {e+1} \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# 	train(train_dataloader, model, loss_fn, optimizer)
# 	print ('\n')

# torch.save(model.state_dict(), 'trained_models/fmnist_bench_pytorch.pth')

model.load_state_dict(torch.load('trained_models/fmnist_bench_pytorch.pth'))

adversarial_test(test_dataloader, model)








































