"""
AlexNet_sequential.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Classic convolutional neural network 'AlexNet'
that pioneered this cNNs for image recognition.
Architecture is implemented using Keras Sequential
model.  Note that the last layer here is performing
binary classification, and does not have the 1000
softmax neurons used in the original network.
"""

### Libraries
# Standard library
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pathlib

# Third-party libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 


### Optimize CPU resource allocation (optional)

AUTOTUNE = tf.data.experimental.AUTOTUNE

### Initialize datasets: all training and test dataset directories are assigned to variables. 
### Insert the correct directory for each desired dataset.

data_dir = pathlib.Path('/home/bbadger/Desktop/NN_snf7',  fname='Combined')
data_dir2 = pathlib.Path('/home/bbadger/Desktop/NN_snf7_2', fname='Combined')
data_dir3 = pathlib.Path('/home/bbadger/Desktop/NN_snf7_3', fname='Combined')

### Assigns the size of the dataset in data_dir to the variable image_count, which is then 
### used to determine the BATCH_SIZE argument for the image_generator.flow_from_directory() function.

image_count = len(list(data_dir.glob('*/*.png')))

### List comprehension to make an array of class names for each image of the given dataset.  The
### class name is determined by the name of the subfolder the image is located inside, eg all
### images inside documents/nn_images/control are labelled 'control', with phantom folders excluded.

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name not in ['._.DS_Store', '._DS_Store', '.DS_Store']])

### Print out the class names to make sure that phantom classes are not included. '._.DS_Store' 
### or variations on this are common names for such classes of size 0.

print (CLASS_NAMES)

### Rescale image bit depth to 8 (if image is 12 or 16 bits) and resize images to 256x256, if necessary

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
IMG_HEIGHT, IMG_WIDTH = 256, 256

### Determine a batch size, ie the number of image per training epoch
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 400


### Keras data generator: assigns the iterable output of image_generator.flow_from_directory() to the 
### variable train_data_gen1, which denotes that this is a training set generated via Keras preprocessing.  
### Most arguments are already specified.  'shuffle' indicates whether or not to randomize images between
### batches (ie between epochs) and 'subset' clarifies which image set is being used (but is not strictly
### necessary)

train_data_gen1 = image_generator.flow_from_directory(directory=str(data_dir),
	batch_size=BATCH_SIZE, shuffle=True, target_size=(IMG_HEIGHT,IMG_WIDTH), 
		classes=list(CLASS_NAMES), subset = 'training')

### Repeat initialization of the Keras data generator for the remaining datasets

CLASS_NAMES = np.array([item.name for item in data_dir2.glob('*') if item.name not in ['._.DS_Store', '.DS_Store', '._DS_Store']])

print (CLASS_NAMES)

test_data_gen1 = image_generator.flow_from_directory(directory=str(data_dir2), 
    batch_size=783, shuffle=True, target_size=(IMG_HEIGHT,IMG_WIDTH),
        classes=list(CLASS_NAMES))


CLASS_NAMES = np.array([item.name for item in data_dir3.glob('*') if item.name not in ['._.DS_Store', '.DS_Store', '._DS_Store']])

print (CLASS_NAMES)

test_data_gen2 = image_generator.flow_from_directory(directory=str(data_dir3), 
    batch_size=719, shuffle=True, target_size=(IMG_HEIGHT,IMG_WIDTH),
        classes=list(CLASS_NAMES))

### assigns training and test image data as the output of Keras preprocessing
### generators

(x_train, y_train) = next(train_data_gen1)
(x_test1, y_test1) = next(test_data_gen1)
(x_test2, y_test2) = next(test_data_gen2)

model = tf.keras.models.Sequential([
    Conv2D(96, (11, 11), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), strides=(4,4), kernel_regularizer=regularizers.l2(0.0005)),
    MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    layers.BatchNormalization(),
    Conv2D(256, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    layers.BatchNormalization(),
    Conv2D(384, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    Conv2D(384, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    Flatten(),
    Dropout(0.5),
    Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    Dropout(0.5),
    Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.0005))
])

### Compile model: choose loss and optimization functions

model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9, lr=0.01, decay=0.3), 
	loss = 'categorical_crossentropy', 
	metrics=['accuracy'])

### Displays details of each layer output in the model 

model.summary()

### Trains the neural network, and print the progress at the end of each epoch
### (signified by verbose=2)

model.fit(x_train, y_train, epochs=9, batch_size = 20, verbose=2)

### Evaluates neural network on test datasets and print the results

model.evaluate(x_test1, y_test1, verbose=2)
model.evaluate(x_test2, y_test2, verbose=2)
