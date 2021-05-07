"""
fmnist_bench.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementation of convolutional neural nets using Keras front-end and a
Tensorflow backend.  Slightly modified versions of Deep_network.py and 
AlexNet.py architectures are employed to classify fashion MNIST images.
"""

### Libraries
# Standard library
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pathlib

# Third-party libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 

import tensorflow as tf

# Helper libraries
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
		Conv2D(16, (3, 3), 
			padding='same', 
			activation='relu', 
			input_shape=(28, 28, 1), 
			data_format='channels_last'),
		MaxPooling2D(),
		Conv2D(16, 3, 
			padding='same', 
			activation='relu'),
		MaxPooling2D(),
		Conv2D(16, 3, 
			padding='same', 
			activation='relu'),
		MaxPooling2D(),
		Conv2D(32, 3, 
			padding='same', 
			activation='relu'),
		MaxPooling2D(),
		Conv2D(32, 3, 
			padding='same', 
			activation='relu'),
		Conv2D(32, 3, 
			padding='same', 
			activation='relu'),
		# MaxPooling2D(),
		Conv2D(64, 3, 
			padding='same', 
			activation='relu'),
		# MaxPooling2D(),
		Conv2D(64, 3, 
			padding='same', 
			activation='relu'),
		# MaxPooling2D(),
		Conv2D(64, 3, 
			padding='same', 
			activation='relu'),
		# MaxPooling2D(),
		Flatten(),
		Dense(512, activation='relu'),
		Dense(50, activation='relu'),
		Dense(10, activation='softmax')
	])

	model.compile(optimizer='Adam', 
		loss = 'sparse_categorical_crossentropy', # labels are integers
		metrics=['accuracy'])

	model.summary()

	model.fit(train_images, train_labels, epochs=20, batch_size = 20, verbose=1)

	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)


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
		loss = 'sparse_categorical_crossentropy', # labels are integers
		metrics=['accuracy'])

	model.summary()

	model.fit(train_images, train_labels, epochs=20, batch_size = 20)

	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

superdeep_network(train_images, train_labels, test_images, test_labels)

alexnet(train_images, train_labels, test_images, test_labels)

'''
##################################
### Optional: displays training or test dataset images with classificaiton labels.
### Note that doing so affects the BATCH_SIZE variable, such that the maximum
### batch size = image_count / 2

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
		plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
		plt.axis('off')
	plt.show()

### calls show_batch on a preprocessed dataset to view classified images

image_batch, label_batch = next(train_data_gen1)
show_batch(image_batch, label_batch)

###################################
'''

### Creates a panel of images classified by the trained neural network.

# image_batch, label_batch = next(test_data_gen1)

# test_images, test_labels = image_batch, label_batch


# def plot_image(i, predictions, true_label, img):
# 	""" returns a test image with a predicted class, prediction
# 	confidence, and true class labels that are color coded for accuracy.
# 	"""
# 	prediction, true_label, img = predictions[i], true_label[i], img[i]
# 	plt.grid(False)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.imshow(img)
# 	predicted_label = np.argmax(predictions)
# 	if prediction[0] >=0.5 and true_label[0]==1:
# 		color = 'green'
# 	elif prediction[0] <=0.5 and true_label[0]==0:
# 		color = 'green'
# 	else:
# 		color = 'red'
# 	plt.xlabel("{} % {}, {}".format(int(100*np.max(prediction)), 
# 		'Snf7' if prediction[0]>=0.5 else 'Control', 
# 		'Snf7' if true_label[0]==1. else 'Control'), color = color)

# 	num_rows = 4
# 	num_cols = 3
# 	num_images = 24

# 	plt.figure(figsize = (num_rows, num_cols))

# 	for i in range(num_images):
# 	  plt.subplot(num_rows, 2*num_cols, i+1)
# 	  plot_image(i+1, predictions, test_labels, test_images)

# 	plt.show() 







