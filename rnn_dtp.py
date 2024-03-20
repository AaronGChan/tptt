import numpy as np
import sys
import argparse
from collections import OrderedDict

from tempOrder import TempOrderTask
from addition import AddTask
from permutation import PermTask
from tempOrder3bit import TempOrder3bitTask

import torch
from torch import nn

# Set default PyTorch data type
torch.set_default_tensor_type(torch.FloatTensor)

def vanilla_sgd(params, grads, learning_rate):
    """
    Update rules for vanilla SGD.

    The update is computed as

        param := param - learning_rate * gradient

    Parameters
    ----------
    params        : list of PyTorch tensors that will be updated
    grads         : list of PyTorch tensors containing the gradients
    learning_rate : step size

    Returns
    -------
    A list of PyTorch tensors containing the updated parameters
    """
    updates = []
    for param, grad in zip(params, grads):
        updates.append(param - learning_rate * grad)

    return updates

def nesterov_momentum(params, grads, learning_rate, momentum=0.9):
    """
    Update rules for Nesterov accelerated gradient descent.

    The update is computed as

        velocity[t] := momentum * velocity[t-1] + learning_rate * gradient[t-1]
        param       := param[t-1] + momentum * velocity[t]
                                  - learning_rate * gradient[t-1]

    Parameters
    ----------
    params        : list of PyTorch tensors that will be updated
    grads         : list of PyTorch tensors containing the gradients
    learning_rate : step size
    momentum      : amount of momentum

    Returns
    -------
    A list of PyTorch tensors containing the updated parameters with applied momentum
    """
    updates = []
    velocities = [torch.zeros_like(param) for param in params]

    for param, grad, velocity in zip(params, grads, velocities):
        update = momentum * velocity + learning_rate * grad
        updates.append(param - update)
        velocities.append(update)

    return updates

def mse(x, y):
    """
    Computes the mean squared error. The average is performed element-wise
    along the squared difference, returning a single value.

    Parameters
    ----------
    x, y : PyTorch tensors

    Returns
    -------
    MSE(x, y)
    """
    return torch.mean((x - y) ** 2)

def rand_ortho(shape, irange, rng):
    """
    Generates an orthogonal matrix. Original code from

    Lee, D. H. and Zhang, S. and Fischer, A. and Bengio, Y., Difference
    Target Propagation, CoRR, abs/1412.7525, 2014

    https://github.com/donghyunlee/dtp

    Parameters
    ----------
    shape  : matrix shape
    irange : range for the matrix elements
    rng    : RandomState instance, initiated with a seed

    Returns
    -------
    An orthogonal matrix of size *shape*
    """
    A = irange * (2 * torch.rand(shape) - 1)
    U, _, V = torch.svd(A)
    return torch.mm(U, V.t())

def sample_length(min_length, max_length, rng):
    """
    Computes a sequence length based on the minimal and maximal sequence size.

    Parameters
    ----------
    max_length      : maximal sequence length (t_max)
    min_length      : minimal sequence length

    Returns
    -------
    A random number from the max/min interval
    """
    length = min_length

    if max_length > min_length:
        length = min_length + rng.randint(max_length - min_length)

    return length

def gaussian(shape, std):
    """
    Draw random samples from a normal distribution.

    Parameters
    ----------
    shape      : output shape
    std        : standard deviation

    Returns
    -------
    Drawn samples from the parameterized normal distribution
    """
    return torch.randn(shape) * std
