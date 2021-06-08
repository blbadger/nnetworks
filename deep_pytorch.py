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
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt  
import torchvision.io


# dataset directory specification
data_dir = pathlib.Path('data/snap29_mono_train1',  fname='Combined')
data_dir2 = pathlib.Path('data/snap29_mono_test1', fname='Combined')
data_dir3 = pathlib.Path('data/snap29_mono_test2', fname='Combined')

image_count = len(list(data_dir.glob('*/*.png')))

class_names = [item.name for item in data_dir.glob('*') 
			   if item.name not in ['._.DS_Store', '._DS_Store', '.DS_Store']]

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
		self.image_name_ls = images[:400]

		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.image_name_ls)

	def __getitem__(self, index):
		# path to image
		img_path = os.path.join(self.image_name_ls[index])
		image = torchvision.io.read_image(img_path) # convert image to tensor of ints in range [0, 255]
		image = image / 255. # convert ints to floats

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
batch_size = 20

train_data = ImageDataset(data_dir)
test_data = ImageDataset(data_dir2)
test_data2 = ImageDataset(data_dir3)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

# deep network specification
class DeepNetwork(nn.Module):

	def __init__(self):
		super().__init__()
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

		self.d1 = nn.Linear(64, 512)
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


model = DeepNetwork() # .to(device)
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters())

def train(dataloader, model, loss_fn, optimizer):
	size = 20
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

		# zero out gradients and backpropegate loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# print (f"Minibatch loss: {loss}")

	ave_loss = float(total_loss) / count
	elapsed_time = time.time() - start
	print (f"Average Loss: {ave_loss:.04}")
	print (f"Completed in {int(elapsed_time)} seconds")
	start = time.time()


def test(dataloader, model):
	size = len(dataloader.dataset)	
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for x, y in dataloader:
			x, y = x.to(device), y.to(device)
			pred = model(x)
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	accuracy = correct / size
	print (f"Test accuracy: {int(correct)} / {size}")
	model.train()

epochs = 20
for e in range(epochs):
	print (f"Epoch {e+1} \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	train(train_dataloader, model, loss_fn, optimizer)
	test(test_dataloader, model)
	print ('\n')

# test(test_dataloader, model)






































