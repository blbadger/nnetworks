"""
Deep_network_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementation of a Keras functional model-based neural network using a 
Tensorflow backend, used for benchmarking the fashion MNIST dataset.
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
print (train_images[:1])


### Neural network model: specifies the architecture using the functional Keras model:
### Conv2D(16, 3, padding='same', activation='relu' signifies a convolutional layers
### of 16 filters, a local receptive field of 3x3, padding such that the output dimensions
### are equal to input dimension (ie 256x256 input --> 256x256 output), and activation
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

# loss_function = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam()
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

# @tf.function
# def train(images, labels):
#     with tf.GradientTape() as tape:
#         predictions = model(images, training=True)
#         loss = loss_function(labels, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     train_loss(loss)
#     train_accuracy(labels, predictions)

# train(train_images, train_labels)

# # model.summary()
model.compile(optimizer='Adam', 
    loss = 'sparse_categorical_crossentropy', # labels are hot-encoded
    metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=9, batch_size = 20, verbose=2)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print (f"Test loss: {test_loss}, \n Test Accuracy: {test_acc}")


image_batch, label_batch = test_images[:25], test_labels[:25]
predictions = deep_model.predict(test_images[:25])

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

plt.show() 








