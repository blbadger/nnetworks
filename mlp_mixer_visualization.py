#! vit_input_visualization.py

# import standard libraries
import time
import pathlib
import os
import pandas as pd 
import random

# import third party libraries
# import seaborn as sns
import sklearn.decomposition as decomp
import numpy as np 
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import mlp_mixer_pytorch 


# dataset directory specification
data_dir = pathlib.Path('tesla',  fname='Combined')

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
        image = torchvision.transforms.Resize(size=[224, 224])(image)

        # assign label
        label = os.path.basename(img_path)
        return image, label

from functools import partial 
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(x) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor=4, expansion_factor_token=0.5, dropout = 0.):
    pair = lambda x: x if isinstance(x, tuple) else (x, x)
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        # nn.LayerNorm(dim),
        # Reduce('b n c -> b c', 'mean'),
        # nn.Linear(dim, num_classes)
    )


images = ImageDataset(data_dir, image_type='.jpg')
model = MLPMixer(
    image_size = 224,
    channels = 3, 
    patch_size = 16,
    dim = 1024,
    depth = 1,
    num_classes = 1000
).to(device)

new_vision = model
new_vision.eval()

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def loss_gradient(model, input_tensor, desired_output, output_dim):
	"""
	 Computes the gradient of the input wrt. the objective function

	 Args:
		input: torch.Tensor() object of input
		model: Transformer() class object, trained neural network

	 Returns:
		gradientxinput: arr[float] of input attributions

	"""

	# change output to float
	desired_output = desired_output.reshape(1)
	input_tensor.requires_grad = True
	output = model.forward(input_tensor.to(device))
	loss = -torch.nn.CrossEntropyLoss()(output, desired_output.to(device))
	# backpropegate output gradient to input
	loss.backward(retain_graph=True)
	gradient = input_tensor.grad
	return gradient

def generate_adversaries(model, single_input, output_tensor, index, count):
	"""
	Plots adversarial examples by applying the gradient of the loss with respect to the input.

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int, example number
		count: timestep number

	returns:
		None (saves .png image)
	"""

	input_grad = loss_gradient(model, single_input, output_tensor, 0)
	added_input = single_input + 0.1*input_grad
	original_pred = model(single_input)

	input_img = single_input.reshape(3, 224, 224).permute(1, 2, 0).cpu().detach().numpy()
	gradient = input_grad * 2
	gradient = gradient.reshape(3, 224, 224).permute(1, 2, 0).cpu().detach().numpy()

	plt.figure(figsize=(18, 10))
	ax = plt.subplot(1, 2, 1)
	plt.axis('off')
	plt.imshow(input_img, alpha=1)
	ax = plt.subplot(1, 2, 2)
	plt.axis('off')
	plt.imshow(gradient, alpha=1)
	plt.tight_layout()
	plt.show()
	plt.close()
	return added_input

for i, image in enumerate(images):
    print (i)
    image = image[0].reshape(1, 3, 224, 224).to(device)
    target_tensor = new_vision(image)
    print (target_tensor.shape)
    break

target_tensor = target_tensor.detach().to(device)
# plt.figure(figsize=(10, 10))
image_width = len(image[0][0])
target_input = image.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
# plt.axis('off')
# plt.imshow(target_input)
# plt.show()
# plt.close()

modification = torch.randn(1, 3, 224, 224)/18
modification = modification.to(device)
modified_input = image + modification
modified_output = new_vision(modified_input)
print (f'L2 distance between original and shifted inputs: {torch.sqrt(torch.sum((image - modified_input)**2))}')
print (f'L2 distance between target and slightly modified image: {torch.sqrt(torch.sum((target_tensor - modified_output)**2))}')

# plt.figure(figsize=(10, 10))
image_width = len(modified_input[0][0])
modified_input = modified_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
# plt.axis('off')
# plt.imshow(modified_input)
# plt.show()
# plt.close()

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
    iterations_arr, input_distances, output_distances = [], [], []
    for i in range(iterations):
        if crop:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
        else:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), len(single_input[0][0]))
            size = len(single_input[0][0])
        single_input = single_input.detach() # remove the gradient for the input (if present)
        input_grad = layer_gradient(new_vision, cropped_input, target_output) # compute input gradient
        single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)*input_grad # gradient descent step
        # single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] = torchvisi on.transforms.functional.gaussian_blur(single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size], 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations))

    return single_input


def generate_singleinput(model, input_tensors, output_tensors, index, count, target_input, random_input=True):
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

    # manualSeed = 999
    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)

    class_index = index
 
    input_distances = []
    iterations_arr = []
    if random_input:
        single_input = (torch.randn(1, 3, 224, 224))/20 + 0.7 # scaled normal distribution initialization

    else:
        single_input = input_tensors[0]

    iterations = 1500
    single_input = single_input.to(device)
    single_input = single_input.reshape(1, 3, 224, 224)
    original_input = torch.clone(single_input).reshape(3, 224, 224).permute(1, 2, 0).cpu().detach().numpy()
    target_output = torch.tensor([class_index], dtype=int)

    single_input = octave(single_input, target_output, iterations, [0.1, 0.1], [2.4, 0.4], 0, pad=False, crop=False)

    output = model(single_input).to(device)
    print (f'L2 distance between target and generated image: {torch.sqrt(torch.sum((target_tensor - output)**2))}')
    target_input = torch.tensor(target_input).reshape(1, 3, 224, 224).to(device)
    input_distance = torch.sqrt(torch.sum((single_input - image)**2))
    print (f'L2 distance on the input: {input_distance}')
    input_distances.append(float(input_distance))
    iterations_arr.append(iterations)

    print (iterations_arr)
    print (input_distances)
    plt.figure(figsize=(10, 10))
    image_width = len(single_input[0][0])
    target_input = single_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
    plt.axis('off')
    plt.imshow(target_input)
    plt.savefig('fig', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return 


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
    output = model(input_tensor)
    loss = 0.06*torch.sum(torch.abs(target_tensor - output)) # target_tensor is the desired activation
    loss.backward()
    gradient = input_tensor.grad

    return gradient


generate_singleinput(new_vision, [], [], 0, 0, image)

