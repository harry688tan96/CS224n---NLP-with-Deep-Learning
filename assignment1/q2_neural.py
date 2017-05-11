#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z2 = np.dot(data, W1) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W2) + b2
    a3 = softmax(z3)
    
    index_of_one = np.argmax(labels, axis=1) #  Each row of the matrix 'labels' is the one-hot row vector. There is only one "1" in each row, the rest are all zeros.
    dimension_of_labels = labels.shape[0]
    cost = - np.sum(np.log(a3[np.arange(dimension_of_labels), index_of_one]))  # Cross-entropy cost
    ### END YOUR CODE

    ### Debugging for foward propagation
    '''
    print "Index of '1' on each row in the matrix 'labels':\n", index_of_one
    print "Probability matrix: \n", a3
    print "Vector corresponding to the '1' in the matrix 'label': \n", a3[np.arange(dimension_of_labels), index_of_one]
    '''
    ### END DEBUG

    ### YOUR CODE HERE: backward propagation
    d3 = a3  # d3 stands for delta_3 (the error at layer 3)
    d3[np.arange(dimension_of_labels), index_of_one] -= 1
    d2 = np.dot(d3, W2.T) * sigmoid_grad(a2)

    gradW2 = np.dot(a2.T, d3)
    gradb2 = np.sum(d3, axis=0)
    gradW1 = np.dot(data.T, d2)
    gradb1 = np.sum(d2, axis=0) 
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    print "You did not implement sanity checks!!"
    # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
