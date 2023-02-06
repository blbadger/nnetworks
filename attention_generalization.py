# cifar10_generalization.py
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
from torch import nn, einsum
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt  
import torchvision.transforms as transforms
from einops import rearrange

# helpers
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    # residual connection

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**(-0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        out = self.maxpool_conv(x)
        return torch.randn(out.shape).to(device)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.attention = Attention(out_channels).to(device)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x2 = self.attention(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

transform = transforms.Compose(
    [transforms.ToTensor()]
    )

batch_size = 64
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**(-0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class TrainedAttention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 3))
        self.down1 = (Down(3, 6))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(6, 3, bilinear))
        self.outc = (OutConv(3, n_classes))
        self.attention = Attention(3).to(device)

    def forward(self, x):
        return self.attention(x)


class AttentionNet(nn.Module):

    def __init__(self, channels):
        super().__init__()
        attention_model = TrainedAttention(3, 3)
        attention_model.load_state_dict(torch.load('../cifar_generation/nonlinear_attention_64_3.pth')) 
        fc_model = SingleEncoder(20000, 3)
        # fc_model.load_state_dict(torch.load('../cifar_generation/trained_models/single_encoder_cifar.pth'))
        self.trained_fc = fc_model
        self.trained_attention = attention_model
        self.attention1 = LinearAttention(channels).to(device)
        self.attention2 = Attention(channels).to(device)
        self.attention3 = Attention(channels).to(device)
        self.fc = nn.Linear(3*32*32, 10)

    def forward(self, x):   
        # x = self.attention1(x)
        # x = self.attention2(x)
        # x = self.attention3(x)
        x = self.trained_attention(x)
        # x = self.trained_fc(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class SingleEncoder(nn.Module):

    def __init__(self, starting_size, channels):
        super().__init__()
        starting = starting_size
        self.input_transform = nn.Linear(32*32*3, starting)
        self.d5 = nn.Linear(starting, 32*32*3)
        self.gelu = nn.GELU()
        self.channels = channels
        self.image_size = 32

    def forward(self, input_tensor):
        input_tensor = torch.flatten(input_tensor, start_dim=1)
        out = self.input_transform(input_tensor)
        out = self.gelu(out)

        out = self.d5(out)
        out = out.reshape(batch_size, self.channels, self.image_size, self.image_size)
        return out

class FCnet(nn.Module):

    def __init__(self, starting_size):

        super().__init__()
        starting = starting_size
        self.input_transform = nn.Linear(32*32*3, starting)
        self.d1 = nn.Linear(starting, starting//2)
        self.d2 = nn.Linear(starting//2, starting//4)
        self.d3 = nn.Linear(starting//4, starting//8)
        self.d4 = nn.Linear(starting//8, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, input_tensor):
        input_tensor = torch.flatten(input_tensor, start_dim=1)
        out = self.input_transform(input_tensor)
        out = self.relu(out)

        out = self.d1(out)
        out = self.relu(out)

        out = self.d2(out)
        out = self.relu(out)

        out = self.d3(out)
        out = self.relu(out)

        out = self.d4(out)
        return out

class ShallowFCnet(nn.Module):
    def __init__(self, starting_size):

        super().__init__()
        starting = starting_size
        self.input_transform = nn.Linear(32*32*3, starting)
        self.d1 = nn.Linear(starting, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, input_tensor):
        input_tensor = torch.flatten(input_tensor, start_dim=1)
        out = self.input_transform(input_tensor)
        out = self.relu(out)
        out = self.d1(out)
        out = self.softmax(out)
        return out

    
class ConvForward(nn.Module):

    def __init__(self, starting_size):

        super().__init__()
        starting = starting_size
        self.conv1 = nn.Conv2d(3, 60, 5, padding=2)
        self.conv2 = nn.Conv2d(60, 80, 5, padding=2)
        self.conv3 = nn.Conv2d(80, 160, 3, padding=1)
        self.conv4 = nn.Conv2d(160, 320, 3, padding=1)
        self.dense = nn.Linear(20480, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.softmax = nn.Softmax()

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = torch.flatten(out, start_dim=1)
        out = self.dense(out)
        out = self.softmax(out)
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
    for n in range(16*16):
        ax = plt.subplot(16, 16, n+1)
        plt.axis('off')
        if grayscale:
            plt.imshow(input_batch[n], cmap='gray_r')
        else:
            plt.imshow(input_batch[n])
        plt.tight_layout()

    plt.tight_layout()
    plt.show()
    plt.savefig('gan_set{0:04d}.png'.format(count), dpi=410)
    plt.close()
    return

def train_model(dataloader, model, optmizer, loss_fn, epochs):
    model.train()
    count = 0
    total_loss = 0
    start = time.time()
    train_array, test_array = [], []

    for e in range(epochs):
        # print (f"Epoch {e+1} \n" + '~'*100)
        total_loss = 0
        count = 0

        for i, pair in enumerate(trainloader):
            if i > 100:
                break
            train_x, train_y = pair[0], pair[1]
            count += 1
            trainx = train_x.to(device)
            output = model(trainx)
            loss = loss_fn(output.to(device), train_y.to(device))
            loss = loss.to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        ave_loss = float(total_loss) / count
        elapsed_time = time.time() - start
        print (f"Epoch {e} complete: Average Loss: {ave_loss:.04}")
        print ('Train: ')
        train_array.append(test_model(trainloader, model))
        print ('Test: ')
        test_array.append(test_model(testloader, model))
        start = time.time()
    print (train_array, test_array)

    return

@torch.no_grad()
def test_model(test_dataloader, model):
    model.eval()
    correct, count = 0, 0
    batches = 0
    for batch, (x, y) in enumerate(test_dataloader):
        if batches > 10:
            break
        x = x.to(device)
        predictions = model(x)
        _, predicted = torch.max(predictions.data, 1)
        count += len(y)
        correct += (predicted == y.to(device)).sum().item()
        batches += 1

    print (f'Accuracy: {correct / count}')
    return correct / count




train_accuracies, test_accuracies = [], []
torch.cuda.empty_cache()
epochs = 50
loss_fn = nn.CrossEntropyLoss()
fcnet = AttentionNet(3)
total_params = sum(p.numel() for p in fcnet.parameters() if p.requires_grad)
fcnet = fcnet.to(device)
# optimizer = torch.optim.Adam(fcnet.parameters(), lr=0.001)
optimizer = torch.optim.SGD(fcnet.parameters(), lr=0.01)
# optimizer2 = torch.optim.Adam(fcnet.parameters(), lr=0.0002)
train_model(trainloader, fcnet, optimizer, loss_fn, epochs)
trainloader = trainloader
testloader = testloader
train_accuracies.append(test_model(trainloader, fcnet))
test_accuracies.append(test_model(testloader, fcnet))

print (train_accuracies)
print (test_accuracies)

