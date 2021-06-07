"""
Deep_network.py
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np 
import matplotlib.pyplot as plt 


### Optimize CPU resource allocation (optional)

AUTOTUNE = tf.data.experimental.AUTOTUNE

### Initialize datasets: all training and test dataset directories are assigned to variables. 
### Insert the correct directory for each desired dataset.

# Snap29 monochrome dataset
data_dir = pathlib.Path('data/snap29_mono_test2',  fname='Combined')
data_dir2 = pathlib.Path('data/snap29_mono_train1', fname='Combined')
data_dir3 = pathlib.Path('data/snap29_mono_test1', fname='Combined') 

# Snf7 dataset
# data_dir = pathlib.Path('data/NN_snf7',  fname='Combined')
# data_dir2 = pathlib.Path('data/NN_snf7_2', fname='Combined')
# data_dir3 = pathlib.Path('data/NN_snf7_2', fname='Combined') # duplicate of above, as there is only one test dataset for Snf7

### Assigns the size of the dataset in data_dir to the variable image_count, which is then 
### used to determine the BATCH_SIZE argument for the image_generator.flow_from_directory() function.

image_count = len(list(data_dir.glob('*/*.png')))

### List comprehension to make an array of class names for each image of the given dataset.  The
### class name is determined by the name of the subfolder the image is located inside, eg all
### images inside documents/nn_images/control are labelled 'control', with phantom folders excluded.

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') 
                        if item.name not in ['._.DS_Store', '._DS_Store', '.DS_Store']])

print (CLASS_NAMES)

### Rescale image bit depth to 8 (if image is 12 or 16 bits) and resize images to 256x256, if necessary

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
IMG_HEIGHT, IMG_WIDTH = 256, 256

### Determine a batch size, ie the number of image per training epoch
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 400

train_data_gen1 = image_generator.flow_from_directory(directory=str(data_dir),
	batch_size=BATCH_SIZE, shuffle=True, target_size=(IMG_HEIGHT,IMG_WIDTH), 
	classes=list(CLASS_NAMES), subset = 'training')

CLASS_NAMES = np.array([item.name for item in data_dir2.glob('*') 
                        if item.name not in ['._.DS_Store', '.DS_Store', '._DS_Store']])
print (CLASS_NAMES)

test_data_gen1 = image_generator.flow_from_directory(directory=str(data_dir2), 
    batch_size=783, shuffle=True, target_size=(IMG_HEIGHT,IMG_WIDTH),
    classes=list(CLASS_NAMES))


CLASS_NAMES = np.array([item.name for item in data_dir3.glob('*') 
                        if item.name not in ['._.DS_Store', '.DS_Store', '._DS_Store']])
print (CLASS_NAMES)

test_data_gen2 = image_generator.flow_from_directory(directory=str(data_dir3), 
    batch_size=719, shuffle=True, target_size=(IMG_HEIGHT,IMG_WIDTH),
    classes=list(CLASS_NAMES))

##################################
### Optional: displays training or test dataset images with classificaiton labels.
### Note that doing so affects the BATCH_SIZE variable strangely

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

### assigns training and test image data as the output of Keras preprocessing
### generators

(x_train, y_train) = next(train_data_gen1)
(x_test1, y_test1) = next(test_data_gen1)
(x_test2, y_test2) = next(test_data_gen2)

### Neural network model: specifies the architecture using the Sequential Keras model:
### Conv2D(16, 3, padding='same', activation='relu' signifies a convolutional layers
### of 16 filters, a local receptive field of 3x3, padding such that the output dimensions
### are equal to input dimension (ie 256x256 input --> 256x256 output), and activation
### function for the layer)

model = tf.keras.models.Sequential([
    Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH , 3)),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='Adam', 
	loss = 'binary_crossentropy',
	metrics=['accuracy'])

model.summary()
model.save_weights('model_init')
accuracies = []

for i in range(10):
    model.load_weights('model_init')
    model.fit(x_train, y_train, epochs=9, batch_size=20, verbose=1)

    ### Evaluates neural network on test datasets and print the results
    string1 = model.evaluate(x_test1, y_test1, verbose=2)
    string2 = model.evaluate(x_test2, y_test2, verbose=2)
    accuracies.append(string1[1])
    accuracies.append(string2[1])

print (accuracies)

### Creates a panel of images classified by the trained neural network.
image_batch, label_batch = next(test_data_gen1)
test_images, test_labels = image_batch, label_batch
predictions = model.predict(test_images)

def plot_image(i, predictions, true_label, img):
    """ returns a test image with a predicted class, prediction
    confidence, and true class labels that are color coded for accuracy.
    """
    prediction, true_label, img = predictions[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    predicted_label = np.argmax(predictions)
    if prediction[0] >=0.5 and true_label[0]==1:
        color = 'green'
    elif prediction[0] <=0.5 and true_label[0]==0:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} % {}, {}".format(int(100*np.max(prediction)), 
        'Snf7' if prediction[0]>=0.5 else 'Control', 
        'Snf7' if true_label[0]==1. else 'Control'), color = color)


num_rows = 4
num_cols = 3
num_images = 24

plt.figure(figsize = (num_rows, num_cols))

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, i+1)
  plot_image(i+1, predictions, test_labels, test_images)

plt.show() 







