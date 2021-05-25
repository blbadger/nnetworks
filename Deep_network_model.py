"""
Tensorflow_sequential_deep.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementation of a Keras sequential neural network using a 
Tensorflow backend, combined with modules to display pre- classified
images before the network trains as well as a subset of test images
with classification and % confidence for each image.  The latter script
is optimized for binary classification but can be modified for more than
two classes.
"""

### Libraries
# Standard library
import pathlib

# Third-party libraries
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np 
import matplotlib.pyplot as plt 

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

img_dimensions = [28, 28] # row, col
test_image_number = len(test_images)
train_image_number = len(train_images)
IMG_HEIGHT = 28
IMG_WIDTH = 28

# reshape input data to match Conv2D input argument
train_images = train_images.reshape(train_image_number, img_dimensions[0], img_dimensions[1], 1)
test_images = test_images.reshape(test_image_number, img_dimensions[0], img_dimensions[1], 1)


### Neural network model: specifies the architecture using the functional Keras model:
### Conv2D(16, 3, padding='same', activation='relu' signifies a convolutional layers
### of 16 filters, a local receptive field of 3x3, padding such that the output dimensions
### are equal to input dimension (ie 28x28 input --> 28x28 output), and activation
### function for the layer)

# Note that convolutional layers, once instatiated, save the input/output dimension sizes: 
# one must instatiate separate layers if either input or output is different. 

class DeepNetwork(Model):

    def __init__(self):
        super(DeepNetwork, self).__init__()
        self.entry_conv = Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1), data_format='channels_last')
        self.conv16 = Conv2D(16, 3, padding='same', activation='relu')
        self.conv32 = Conv2D(32, 3, padding='same', activation='relu')
        self.conv32_2 = Conv2D(32, 3, padding='same', activation='relu')
        self.conv64 = Conv2D(64, 3, padding='same', activation='relu')
        self.conv64_2 = Conv2D(64, 3, padding='same', activation='relu')
        self.max_pooling = MaxPooling2D()

        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(50, activation='relu')
        self.d3 = Dense(10, activation='softmax')
        

    def call(self, model_input):
        out = self.entry_conv(model_input)
        for _ in range(2):
            out = self.conv16(out)
            out = self.max_pooling(out)
        out2 = self.conv32(out)
        out2 = self.max_pooling(out2)
        for _ in range(2):
            out2 = self.conv32_2(out2)
        out3 = self.max_pooling(out2)
        out3 = self.conv64(out3)
        for _ in range(2):
            out3 = self.conv64_2(out3)
            # out3 = self.max_pooling(out3)
        output = self.flatten(out3)
        output = self.d1(output)
        output = self.d2(output)
        final_output = self.d3(output)
        return final_output

model = DeepNetwork()

model.compile(optimizer='Adam', 
    loss = 'sparse_categorical_crossentropy', # labels are hot-encoded
    metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=9, batch_size = 20, verbose=2)

### Evaluates neural network on test datasets and print the results

model.evaluate(x_test1, y_test1, verbose=2)

model.evaluate(x_test2, y_test2, verbose=2)

predictions = model.predict(test_images)
### Creates a panel of images classified by the trained neural network.

image_batch, label_batch = test_images[:25], test_labels[:25]
predictions = deep_model.predict(test_images[:25])

def plot_image(image_batch, test_labels, predictions):
    """ 
    Returns a test image with a predicted class, prediction
    confidence, and true class labels that are color coded for accuracy.
    """
    plt.figure(figsize=(20,20))
    for i in range(25):
        ax = plt.subplot(5, 5, i+1)
        plt.imshow(image_batch[i].reshape(28, 28), cmap='gray')
        max_index = np.argmax(predictions[i])
        predicted_label = class_names[np.argmax(predictions[i])]
        if max_index == test_labels[i]:
            color = 'green'
        else:
            color = 'red'
        confidence = int(100 * round(predictions[i][max_index], 2))
        plt.title(f"{confidence} % {predicted_label}, {class_names[test_labels[i]]}", color=color, fontsize=10)
        plt.axis('off')

    plt.show() 

plot_image(image_batch, label_batch, predictions)







