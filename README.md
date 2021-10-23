# SUPERVISED ENCODING FOR DISCRETE REPRESENTATION LEARNING

This is the source code for Supervised Encoding for Discrete Representation Learning paper (https://arxiv.org/pdf/1910.11067.pdf).


## Description

Classical supervised classification tasks search for a nonlinear mapping that maps each encoded feature directly to a probability mass over the labels. Such a learning framework typically lacks the intuition that encoded features from the same class tend to be similar and thus has little interpretability for the learned features. In this paper, we propose a novel supervised learning model named Supervised-Encoding Quantizer (SEQ). The SEQ applies a quantizer to cluster and classify the encoded features. We found that the quantizer provides an interpretable graph where each cluster in the graph represents a class of data samples that have a particular style. We also trained a decoder that can decode convex combinations of the encoded features from similar and different clusters and provide guidance on style transfer between sub-classes.

<p align="center">
  <img src="images/fig1.jpg" width="350" title="Illustration of the supervised-encoding quantizer (SEQ)">
  <img src="images/fig2.jpg" width="350" title=" Illustration of the quantization process">
</p>

## Getting Started

### Dependencies

* Requires Pytorch, Numpy
* MNIST dataset (https://www.kaggle.com/oddrationale/mnist-in-csv)
* fashion-MNIST dataset (https://www.kaggle.com/zalando-research/fashionmnist)

### Executing program

* First, train the encoder of the autoencoder. The hidden features are clustered by k-means quantizer algorithm.
```
python main_encoder.py
```
* Next, train the decoder of the autoencoder. Generate new data using the decoder with the convex hull combination features.
```
python main_decoder_mnist.py
```
Similarly, for fashion-MNIST, run the following:
```
python main_decoder_fmnist.py
```

## Authors

Cat P. Le (cat.le@duke.edu), 
Yi Zhou, 
Jie Ding, 
Vahid Tarokh