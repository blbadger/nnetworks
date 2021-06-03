# deep_pytorch.py

# import standard libraries
import time

# import third party libraries
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt  

# fashion MNIST dataset train and test data load
training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

# specify batch size
batch_size = 20

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

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

        self.d1 = nn.Linear(144, 512)
        self.d2 = nn.Linear(512, 50)
        self.d3 = nn.Linear(50, 10)
        

    def forward(self, model_input):
        out = self.relu(self.entry_conv(model_input))
        out = self.max_pooling(out)
        out = self.relu(self.conv32(out))
        out = self.max_pooling(out)
        out = self.relu(self.conv64(out))
        out = self.max_pooling(out)

        # out = self.relu(self.conv32(out))
        # out = self.relu(self.conv32_2(out))
        # out = self.relu(self.conv32_2(out))

        # out = self.relu(self.conv64(out))
        # out = self.relu(self.conv64_2(out))
        # out = self.relu(self.conv64_2(out))
        # out = self.max_pooling(out)
        output = self.flatten(out)

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
	size = len(dataloader.dataset)
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
			# x, y = x.to(device), y.to(device)
			pred = model(x)
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	accuracy = correct / size
	print (f"Test accuracy: {int(correct)} / {size}")
	model.train()

epochs = 10
for e in range(epochs):
	print (f"Epoch {e+1} \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	train(train_dataloader, model, loss_fn, optimizer)
	test(test_dataloader, model)
	print ('\n')

# test(test_dataloader, model)






































