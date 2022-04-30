# nnetworks

A collection of neural networks for image classification. `interprets` contains interpretable sequence-based deep learning models for tabular data input, and `connected` contains fully-connected neural networks built from scratch that classify MNIST handwritten digit images.

cNN design was inspired by the official Tensorflow Keras introduction (https://www.tensorflow.org/tutorials/keras/classification) and load image (https://www.tensorflow.org/tutorials/load_data/images?hl=TR) tutorial. 

Contains a collection of snippets (NN_prep_snippets) to prepare test and training image datasets for a convolutional neural network (Deep_network) built with a Keras API (both sequential and functional models are presented) on a Tensorflow backend.

Deep_network architecture:
![neural network architecture](https://github.com/blbadger/blbadger.github.io/blob/master/misc_images/cNN_architecture.png)

which accurately predicts cell genotype based on fluorescent microscopic images
![classification](https://github.com/blbadger/blbadger.github.io/blob/master/neural_networks/nn_images_1.png)
