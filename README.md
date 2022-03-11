# Convolutional Neural Network

## Overview

Supervised. Used a lot on image recognition. A ConvNet mimics how mammals visually perceive the world. 
Images are decomposed in smaller shapes, the image is then identified by recognizing the presence of some particular shapes. 
To do so, CNN uses a receptive field (a part of the image that we are focused on (e.g. a square of 10x10 pixels)) 
and slides it across the image. We apply a convolution to every sub-image and then max-pool it to a new matrix.

## Dataset 

Every image is a matrix of pixel values.
The possible range of values a single pixel can represent is [0, 255].
However, with coloured images, particularly RGB (Red, Green, Blue)-based images, the presence of separate colour channels (3 in the case of RGB images) introduces an additional ‘depth’ field to the data, making the input 3-dimensional.
Hence, for a given RGB image of size, say 255×255 (Width x Height) pixels, we’ll have 3 matrices associated with each image, one for each of the colour channels.
Thus the image in it’s entirety, constitutes a 3-dimensional structure called the Input Volume (255x255x3).

### How does it work?

CNN use series of convolution and max-pooling layers. At the end of the net, a fully connected layer with a soft-max
activation function outputs a vector of probabilities associated to every label.

## 1. Convolution

* A convolution is an orderly procedure where two sources of information are intertwined.
* A kernel (also called a filter) is a smaller-sized matrix in comparison to the input dimensions of the image, that consists of real valued entries.
* Kernels are then convolved with the input volume to obtain so-called ‘activation maps’ (also called feature maps).
* Activation maps indicate ‘activated’ regions, i.e. regions where features specific to the kernel have been detected in the input.
* The real values of the kernel matrix change with each learning iteration over the training set, indicating that the network is learning to identify which regions are of significance for extracting features from the data.
* We compute the dot product between the kernel and the input matrix. -The convolved value obtained by summing the resultant terms from the dot product forms a single entry in the activation matrix.
* The patch selection is then slided (towards the right, or downwards when the boundary of the matrix is reached) by a certain amount called the ‘stride’ value, and the process is repeated till the entire input image has been processed. - The process is carried out for all colour channels.
instead of connecting each neuron to all possible pixels, we specify a 2 dimensional region called the ‘receptive field’ (say of size 5×5 units) extending to the entire depth of the input (5x5x3 for a 3 colour channel input), within which the encompassed pixels are fully connected to the neural network’s input layer. It’s over these small regions that the network layer cross-sections (each consisting of several neurons (called ‘depth columns’)) operate and produce the activation map. (reduces computational complexity)

## 2. Pooling

Pooling reducing the spatial dimensions (Width x Height) of the Input Volume for the next Convolutional Layer. It does not affect the depth dimension of the Volume.
The transformation is either performed by taking the maximum value from the values observable in the window (called ‘max pooling’), or by taking the average of the values. Max pooling has been favoured over others due to its better performance characteristics.
also called downsampling

## 3. Normalization (ReLU) and Regularization
 
* **Normalization** (keep the math from breaking by turning all negative numbers to 0) (RELU) a stack of images becomes a stack of images with no negative values.
Repeat Steps 2-4 several times. More, smaller images (feature maps created at every layer)

* **Regularization**
Dropout forces an artificial neural network to learn multiple independent representations of the same data by alternately randomly disabling neurons in the learning phase.
Dropout is a vital feature in almost every state-of-the-art neural network implementation.
To perform dropout on a layer, you randomly set some of the layer’s values to 0 during forward propagation.

## 4. Probability Conversion
At the very end of our network (the tail), we’ll apply a softmax function to convert the outputs to probability values for each class,
and select the most likely label with argmax(softmax_outputs)

## Use cases

* Image, Video, Sound recognition
* Image, Video, Sound generation (flip the ConvNet !)

## Credits

* Siraj Raval
* [greydanus](https://github.com/greydanus/pythonic_ocr)
