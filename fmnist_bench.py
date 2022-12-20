"""
fmnist_bench.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementation of convolutional neural nets using Keras front-end and a
Tensorflow backend.  Slightly modified versions of Deep_network.py and 
AlexNet.py architectures are employed to classify fashion MNIST images.
"""

### Libraries
# Third-party libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras import layers

import numpy as np 
import matplotlib.pyplot as plt 

# Preprocess data

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
			   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

img_dimensions = [28, 28] # row, col
test_image_number = len(test_images)
train_image_number = len(train_images)

'''
### Optional: displays training or test dataset images with classification labels.
def show_batch(image_batch, label_batch):
	"""Takes a set of images (image_batch) and an array of labels (label_batch)
	from the tensorflow preprocessing modules above as arguments and subsequently
	returns a plot of a 25- member subset of the image set specified, complete
	with a classification label for each image.  
	"""
	plt.figure(figsize=(10,10))
	for n in range(25):
		ax = plt.subplot(5, 5, n+1)
		plt.imshow(image_batch[n])
		plt.xlabel(class_names[label_batch[n]==1][0].title())
		plt.axis('off')
	plt.show()

### calls show_batch on a preprocessed dataset to view classified images

image_batch, label_batch = next(train_data_gen1)
show_batch(image_batch, label_batch)
'''

# reshape input data to match Conv2D input argument
train_images = train_images.reshape(train_image_number, img_dimensions[0], img_dimensions[1], 1)
test_images = test_images.reshape(test_image_number, img_dimensions[0], img_dimensions[1], 1)


def superdeep_network(train_images, train_labels, test_images, test_labels):
	### Neural network model: specifies the architecture using the Sequential Keras model:
	### Conv2D(16, 3, padding='same', activation='relu' signifies a convolutional layers
	### of 16 filters, a local receptive field of 3x3, padding such that the output dimensions
	### are equal to input dimension (ie 256x256 input --> 256x256 output), and activation
	### function for the layer)
	model = tf.keras.models.Sequential([
		Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1), data_format='channels_last'),
		MaxPooling2D(),
		Conv2D(16, 3, padding='same', activation='relu'),
		MaxPooling2D(),
		Conv2D(16, 3, padding='same', activation='relu'),
		MaxPooling2D(),
		Conv2D(32, 3, padding='same', activation='relu'),
		# MaxPooling2D(),
		Conv2D(32, 3, padding='same', activation='relu'),
		Conv2D(32, 3, padding='same', activation='relu'),
		# MaxPooling2D(),
		Conv2D(64, 3, padding='same', activation='relu'),
		# MaxPooling2D(),
		Conv2D(64, 3, padding='same', activation='relu'),
		# MaxPooling2D(),
		Conv2D(64, 3, padding='same', activation='relu'),
		MaxPooling2D(),
		Flatten(),
		Dense(512, activation='relu'),
		Dense(50, activation='relu'),
		Dense(10, activation='softmax')
	])

	model.compile(optimizer='Adam', 
		loss = 'sparse_categorical_crossentropy', # labels are integers
		metrics=['accuracy'])

	model.summary()
	model.fit(train_images, train_labels, epochs=10, batch_size=64, verbose=1)
	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
	return model


def alexnet(train_images, train_labels, test_images, test_labels):
	# AlexNet model

	model = tf.keras.models.Sequential([
		Conv2D(96, (11, 11), 
			activation='relu', 
			input_shape=(28, 28, 1), 
			strides=(4,4), 
			kernel_regularizer=regularizers.l2(0.0005)),
		MaxPooling2D(pool_size=(3,3), strides=(2,2)),
		layers.BatchNormalization(),
		Conv2D(256, 5, 
			padding='same', 
			activation='relu', 
			kernel_regularizer=regularizers.l2(0.0005)),
		# MaxPooling2D(pool_size=(3,3), strides=(2,2)),
		layers.BatchNormalization(),
		Conv2D(384, 3, 
			padding='same', 
			activation='relu', 
			kernel_regularizer=regularizers.l2(0.0005)),
		Conv2D(384, 3, 
			padding='same', 
			activation='relu', 
			kernel_regularizer=regularizers.l2(0.0005)),
		Conv2D(256, 3, 
			padding='same', 
			activation='relu', 
			kernel_regularizer=regularizers.l2(0.0005)),
		# MaxPooling2D(pool_size=(3,3), strides=(2,2)),
		Flatten(),
		Dropout(0.5),
		Dense(2048, 
			activation='relu', 
			kernel_regularizer=regularizers.l2(0.0005)),
		Dropout(0.5),
		Dense(2048, 
			activation='relu', 
			kernel_regularizer=regularizers.l2(0.0005)),
		Dense(10, 
			activation='softmax', 
			kernel_regularizer=regularizers.l2(0.0005))
	])

	### Compile model: choose loss and optimization functions

	model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9, lr=0.01, decay=0.3), 
		loss = 'sparse_categorical_crossentropy', # labels are integers, not one-hot tensors
		metrics=['accuracy'])

	model.summary()
	model.fit(train_images, train_labels, epochs=10, batch_size=20)
	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
	return

model = superdeep_network(train_images, train_labels, test_images, test_labels)

# alexnet(train_images, train_labels, test_images, test_labels)

### Creates a panel of images classified by the trained neural network.
image_batch, label_batch = test_images[:25], test_labels[:25]
predictions = model.predict(test_images[:25])

def gradientxinput(features, label):
	optimizer = tf.keras.optimizers.Adam()
	features = features.reshape(1, 28, 28, 1)
	ogfeatures = features
	features = tf.Variable(features, dtype=tf.float32)
	with tf.GradientTape() as tape:
		predictions = model(features)

	input_gradients = tape.gradient(predictions, features).numpy()
	input_gradients = input_gradients.reshape(28, 28)
	ogfeatures = ogfeatures.reshape(28, 28)
	gradxinput = tf.abs(input_gradients) * ogfeatures

	ax = plt.subplot(1, 3, 1)
	plt.axis('off')
	plt.title('Input')
	plt.imshow(ogfeatures, cmap='gray', alpha=1)
	ax = plt.subplot(1, 3, 2)
	plt.axis('off')
	plt.title('Gradient * Input')
	plt.imshow(gradxinput, cmap='inferno', alpha=1)
	ax = plt.subplot(1, 3, 3)
	plt.axis('off')
	plt.title('Combined')
	plt.imshow(ogfeatures, cmap='gray', alpha=1)
	plt.imshow(gradxinput, cmap='inferno', alpha=0.5)
	plt.tight_layout()
	plt.show()
	plt.close()

	return

def plot_image(image, prediction, true_label):
	""" 
	Returns a test image with a predicted class, prediction
	confidence, and true class labels that are color coded for accuracy.
	"""
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(image.reshape(28, 28), cmap='gray')
	max_index = np.argmax(prediction)
	confidence = prediction[max_index]
	predicted_label = class_names[max_index]
	if max_index == true_label:
		color = 'green'
	else:
		color = 'red'
	confidence = int(100 * round(confidence, 2))
	plt.xlabel(f"{confidence} % {predicted_label}, {class_names[true_label]}", color=color)

num_rows = 4
num_cols = 3
num_images = 24

plt.figure(figsize = (num_rows, num_cols))

for i in range(num_images):
	plt.subplot(num_rows, 2*num_cols, i+1)
	plot_image(image_batch[i], predictions[i], test_labels[i])
	gradientxinput(image_batch[i], label_batch[i])


plt.show()

