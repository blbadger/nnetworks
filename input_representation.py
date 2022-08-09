# latent_space.py

# import standard libraries
import time
import pathlib
import os
import pandas as pd 
import random

# import third party libraries
import seaborn as sns
import sklearn.decomposition as decomp
import numpy as np 
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from google.colab import files
# from google.colab import drive

# drive.mount('/content/gdrive')

# data_dir = pathlib.Path('/content/gdrive/My Drive/Inception_generation',  fname='Combined')
# image_count = len(list(data_dir.glob('*.png')))

# files.upload() # upload files
# !unzip dalmatian.zip

# dataset directory specification
data_dir = pathlib.Path('dalmatian',  fname='Combined')

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class ImageDataset(Dataset):
    """
    Creates a dataset from images classified by folder name.  Random
    sampling of images to prevent overfitting
    """

    def __init__(self, img_dir, image_type='.png'):
        self.image_name_ls = list(img_dir.glob('*/*' + image_type))
        self.img_labels = [item.name for item in data_dir.glob('*/*')]
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_name_ls)

    def __getitem__(self, index):
        # path to image
        img_path = os.path.join(self.image_name_ls[index])
        image = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB) # convert image to tensor of ints 
        image = image / 255. # convert ints to floats in range [0, 1]
        image = torchvision.transforms.Resize(size=[299, 299])(image)

        # assign label
        label = os.path.basename(img_path)
        return image, label


googlenet = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True, init_weights=False).to(device)
images = ImageDataset(data_dir, image_type='.jpg')

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).to(device)
googlenet.eval()
resnet.eval()

# network = NewGoogleNet(googlenet)
# network.eval()
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

class NewResNet(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x):
		x = self.model.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)

		x = self.model.layer1(x)
		# x = self.model.layer2(x)
		# x = self.model.layer3(x)
		# x = self.model.layer4(x)

		# x = self.model.avgpool(x)
		# x = self.model.flatten(x, 1)
		# x = self.model.fc(x)
		return x

class NewGoogleNet(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x):
			# N x 3 x 224 x 224
			x = self.model.conv1(x)
			# N x 64 x 112 x 112
			x = self.model.maxpool1(x)
			# N x 64 x 56 x 56
			x = self.model.conv2(x)
			# # N x 64 x 56 x 56
			x = self.model.conv3(x)
			# # N x 192 x 56 x 56
			x = self.model.maxpool2(x)
			# # N x 192 x 28 x 28
			x = self.model.inception3a(x)
			# # N x 256 x 28 x 28
			# x = self.model.inception3b(x)
			# # N x 480 x 28 x 28
			# x = self.model.maxpool3(x)
			# # N x 480 x 14 x 14
			# x = self.model.inception4a(x)
			# N x 512 x 14 x 14
			# x = self.model.inception4b(x)
			# N x 512 x 14 x 14
			# x = self.model.inception4c(x)
			# # N x 512 x 14 x 14
			# x = self.model.inception4d(x)
			# # N x 528 x 14 x 14
			# x = self.model.inception4e(x)
			# # N x 832 x 14 x 14
			# x = self.model.maxpool4(x)
			# # N x 832 x 7 x 7
			# x = self.model.inception5a(x)
			# # N x 832 x 7 x 7
			# x = self.model.inception5b(x)
			# # N x 1024 x 7 x 7
			# x = self.model.avgpool(x)
			# # N x 1024 x 1 x 1
			# x = torch.flatten(x, 1)
			# # N x 1024
			# x = self.model.dropout(x)
			# x = self.model.fc(x)
			# N x 1000 (num_classes)
			return x

resnet2 = NewResNet(resnet)
resnet2.eval()


# for param in resnet2.parameters():
#     print (param)
#     param.data = nn.parameter.Parameter(torch.ones_like(param)/10)
#     print (param)


# googlenet2 = NewGoogleNet(googlenet)
# googlenet2.eval()

# for layer in resnet2.modules():
#     if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d): 
#         print ('Betas: ', layer.bias)
#         print ('Weights: ', layer.weight)

def train(model, input_tensor, target_output):
    """
    Train a single minibatch

    Args:
        input_tensor: torch.Tensor object 
        output_tensor: torch.Tensor object
        optimizer: torch.optim object
        minibatch_size: int, number of examples per minibatch
        model: torch.nn

    Returns:
        output: torch.Tensor of model predictions
        loss.item(): float of loss for that minibatch

    """
    # self.model.train()
    output = model(input_tensor)
    loss = torch.sum(torch.abs(output - target_output)**2)
    optimizer.zero_grad() # prevents gradients from adding between minibatches
    loss.backward()
    optimizer.step()
    return 


optimizer = torch.optim.SGD(resnet.parameters(), lr=0.00001)
for i, image in enumerate(images):
    if i == 0:
        print (i)
        image = image[0].reshape(1, 3, 299, 299).to(device)
        # target_output = googlenet(image).detach().to(device)
        target_tensor = resnet2(image)
        # image2 = torchvision.transforms.Resize(size=[470, 470])(image)
        # target_tensor2 = resnet2(image2).detach().to(device)
        # image3 = torchvision.transforms.Resize(size=[570, 570])(image)
        # target_tensor3 = resnet2(image3).detach().to(device)
        break

target_tensor = target_tensor.detach().to(device)
# resnet.eval()

plt.figure(figsize=(10, 10))
image_width = len(image[0][0])
target_input = image.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
plt.axis('off')
plt.imshow(target_input)
plt.show()
plt.close()

modification = torch.randn(1, 3, 299, 299)/25
modification = modification.to(device)
modified_input = image + modification
modified_output = resnet2(modified_input).to(device)
print (len(modified_output.flatten()))
print (f'L2 distance between target and slightly modified image: {torch.sqrt(torch.sum((target_tensor - modified_output)**2))}')

plt.figure(figsize=(10, 10))
image_width = len(modified_input[0][0])
modified_input = modified_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
plt.axis('off')
plt.imshow(modified_input)
plt.show()
plt.close()

def random_crop(input_image, size):
    """
    Crop an image with a starting x, y coord from a uniform distribution

    Args:
        input_image: torch.tensor object to be cropped
        size: int, size of the desired image (size = length = width)

    Returns:
        input_image_cropped: torch.tensor
        crop_height: starting y coordinate
        crop_width: starting x coordinate
    """

    image_width = len(input_image[0][0])
    image_height = len(input_image[0])
    crop_width = random.randint(0, image_width - size)
    crop_height = random.randint(0, image_width - size)
    input_image_cropped = input_image[:, :, crop_height:crop_height + size, crop_width: crop_width + size]

    return input_image_cropped, crop_height, crop_width


def octave(single_input, target_output, iterations, learning_rates, sigmas, size, pad=False, crop=True):
    """
    Perform an octave (scaled) gradient descent on the input.

    Args;
        single_input: torch.tensor of the input
        target_output: torch.tensor of the desired output category
        iterations: int, the number of iterations desired
        learning_rates: arr[int, int], pair of integers corresponding to start and end learning rates
        sigmas: arr[int, int], pair of integers corresponding to the start and end Gaussian blur sigmas
        size: int, desired dimension of output image (size = length = width)

    kwargs:
        pad: bool, if True then padding is applied at each iteration of the octave
        crop: bool, if True then gradient descent is applied to cropped sections of the input

    Returns:
        single_input: torch.tensor of the transformed input
    """

    start_lr, end_lr = learning_rates
    start_sigma, end_sigma = sigmas

    for i in range(iterations):
        if crop:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
        else:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), len(single_input[0][0]))
            size = len(single_input[0][0])
        single_input = single_input.detach() # remove the gradient for the input (if present)
        input_grad = layer_gradient(resnet, cropped_input, target_output) # compute input gradient
        single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)* input_grad # gradient descent step
        single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] = torchvision.transforms.functional.gaussian_blur(single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size], 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations))

    return single_input


def generate_singleinput(model, input_tensors, output_tensors, index, count, random_input=True):
    """
    Generates an input for a given output

    Args:
        input_tensor: torch.Tensor object, minibatch of inputs
        output_tensor: torch.Tensor object, minibatch of outputs
        index: int, target class index to generate
        cout: int, time step

    kwargs: 
        random_input: bool, if True then a scaled random normal distributionis used

    returns:
        None (saves .png image)
    """

    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    class_index = index

    if random_input:
        single_input = (torch.randn(1, 3, 299, 299))/20 + 0.7 # scaled normal distribution initialization

    else:
        single_input = input_tensors[0]
 
    single_input = single_input.to(device)
    single_input = single_input.reshape(1, 3, 299, 299)
    original_input = torch.clone(single_input).reshape(3, 299, 299).permute(1, 2, 0).cpu().detach().numpy()
    target_output = torch.tensor([class_index], dtype=int)

    single_input = octave(single_input, target_output, 220, [2.5, 1.5], [2.4, 0.4], 0, pad=False, crop=False)

    output = resnet2(single_input).to(device)
    print (f'L2 distance between target and generated image: {torch.sqrt(torch.sum((target_tensor - output)**2))}')

    plt.figure(figsize=(10, 10))
    image_width = len(single_input[0][0])
    target_input = single_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
    plt.axis('off')
    plt.imshow(target_input)
    plt.show()
    plt.close()
    return single_input


def layer_gradient(model, input_tensor, desired_output):
    """
    Compute the gradient of the output (logits) with respect to the input 
    using an L1 metric to maximize the target classification.

    Args:
        model: torch.nn.model
        input_tensor: torch.tensor object corresponding to the input image
        true_output: torch.tensor object of the desired classification label

    Returns:
        gradient: torch.tensor.grad on the input tensor after backpropegation

    """
    input_tensor.requires_grad = True
    output = resnet2(input_tensor)
    loss = 0.01*torch.sum(torch.abs(target_tensor - output)) # target_tensor is the desired activation

    loss.backward()
    gradient = input_tensor.grad

    return gradient

generate_singleinput(resnet2, [], [], 0, 0)