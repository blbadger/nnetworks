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
from google.colab import drive

drive.mount('/content/gdrive')

data_dir = pathlib.Path('/content/gdrive/My Drive/googlenet',  fname='Combined')
image_count = len(list(data_dir.glob('*.png')))

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class ImageDataset(Dataset):
    """
    Creates a dataset from images classified by folder name.  Random
    sampling of images to prevent overfitting
    """

    def __init__(self, img_dir, image_type='.png'):
        self.image_name_ls = list(img_dir.glob('*' + image_type))
        self.img_labels = [item.name for item in data_dir.glob('*')]
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


class NewGoogleNet(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.conv3.bn.apply(blank_batchnorm)
        # self.model.inception5a.branch1.bn.apply(blank_batchnorm)
        # self.model.inception5a.branch2[1].bn.apply(blank_batchnorm)
        # self.model.inception5a.branch3[1].bn.apply(blank_batchnorm)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.model.conv1(x)
        # N x 64 x 112 x 112
        x = self.model.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.model.conv2(x)
        # N x 64 x 56 x 56
        x = self.model.conv3(x)
        # # N x 192 x 56 x 56
        # x = self.model.maxpool2(x)
        # # N x 192 x 28 x 28
        # x = self.model.inception3a(x)
        # # N x 256 x 28 x 28
        # x = self.model.inception3b(x)
        # # N x 480 x 28 x 28
        # x = self.model.maxpool3(x)
        # # N x 480 x 14 x 14
        # x = self.model.inception4a(x)
        # # N x 512 x 14 x 14
        # x = self.model.inception4b(x)
        # # N x 512 x 14 x 14
        # x = self.model.inception4c(x)
        # # N x 512 x 14 x 14
        # x = self.model.inception4d(x)
        # # N x 528 x 14 x 14
        # x = self.model.inception4e(x)
        # # N x 832 x 14 x 14
        # x = self.model.maxpool4(x)
        # # N x 832 x 7 x 7
        # x = self.model.inception5a(x)
        # N x 832 x 7 x 7
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


def blank_batchnorm(layer):
    layer.reset_parameters()
    layer.eval()
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.zero_()
    return

googlenet = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True).to(device)
images = ImageDataset(data_dir, image_type='.png')

# network = NewGoogleNet(googlenet).to(device)
outputs, labels_arr = [], []
for i, image in enumerate(images):
    print (i)
    label = image[1]
    image = image[0].reshape(1, 3, 299, 299).to(device)
    output = googlenet(image)
    output = output.detach().cpu().numpy()
    outputs.append(output)
    i = 11
    while label[i] not in ',.':
        i += 1
    labels_arr.append(label[11:i])

outputs = torch.tensor(outputs)
outputs = outputs.reshape(len(outputs), 1000)
pca = decomp.PCA(n_components=2)
pca.fit(outputs)
print (pca.explained_variance_ratio_)
arr = pca.transform(outputs)
x, y = [i[0] for i in arr], [i[1] for i in arr]
plt.figure(figsize=(18, 18))
plt.scatter(x, y)
for i, label in enumerate(labels_arr):
    plt.annotate(label, (x[i], y[i]))

plt.xlabel('Feature 0')
plt.ylabel('Feature 4')
plt.title('GoogleNet Layer 5a Embedding')
plt.show()
plt.close()

sns.jointplot(x, y)
plt.show()
plt.close()

