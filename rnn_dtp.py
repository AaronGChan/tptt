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
import torch.nn.functional as F

# Set default PyTorch data type
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float32)

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
    updates = OrderedDict() # []
    for param, grad in zip(params, grads):
        # updates.append(param - learning_rate * grad)
        updates[param] = param - learning_rate * grad

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
    updates = vanilla_sgd(params, grads, learning_rate) #[]
    # velocities = [torch.zeros_like(param) for param in params]

    # for param, grad, velocity in zip(params, grads, velocities):
    #     update = momentum * velocity + learning_rate * grad
    #     updates.append(param - update)
    #     velocities.append(update)

    for param in params:
        value = param.clone()
        velocity = torch.nn.Parameter(torch.zeros_like(value), required_grad=False)
        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]


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
    return torch.mm(U, torch.mm(torch.eye(U.shape[1], V.shape[0]), V))

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

def fit(rng, i_learning_rate, f_learning_rate, g_learning_rate, n_hid, init, batch_size, max_length, min_length, task, maxiter, chk_interval, gaussian_noise, val_size, val_batch, gd_opt, task_name, wxh_updates):
    """
    Fits a TPTT-trained SRN model

    Parameters
    ----------
    rng             : RandomState instance, initiated with a seed
    i_learning_rate : initial learning rate (alpha_i)
    f_learning_rate : forward learning rate (alpha_f)
    g_learning_rate : feedback learning rate (alpha_g)
    n_hid           : number of neurons in the hidden layer
    init            : hidden units initialisation
    batch_size      : number of samples per mini-batch
    max_length      : maximal sequence length (t_max)
    min_length      : minimal sequence length 
    task            : task type (addition, temporal order etc.)
    maxiter         : maximal number of iterations
    chk_interval    : number of iterations between validation
    gaussian_noise  : amount of injected Gaussian noise
    val_size        : size of the validation set
    val_batch       : number of samples in the validation set
    gd_opt          : optimisation technique (vanilla sgd, nesterov)
    task_name       : name of the synthetic problem
    wxh_updates     : update mechanism for Wxh
    """

    print("------------------------------------------------------")
    print("******************************************************")
    print("Parameters - Simple RNN TPTT")
    print("******************************************************")
    print("task: %s" % task_name)
    print("optimization: %s" % gd_opt)
    print("i_learning_rate: %f" % i_learning_rate)
    print("f_learning_rate: %f" % f_learning_rate)
    print("g_learning_rate: %f" % g_learning_rate)
    print("maxiter: %i" % maxiter)
    print("batch_size: %i" % batch_size)
    print("min_length: %i" % min_length)
    print("max_length: %i" % max_length)
    print("chk_interval: %i" % chk_interval)
    print("n_hid: %i" % n_hid)
    print("init: %s" % init)
    print("val_size: %i" % val_size)
    print("val_batch: %i" % val_batch)
    print("noise: %f" % gaussian_noise)    
    print("wxh_updates: %s" % wxh_updates)
    print("******************************************************")

    # Get the number of inputs and outputs from the task
    n_inp = task.nin
    n_out = task.nout

    # Initialise the model parameters at random based on the specified
    # activation function 
    if init == "sigmoid":
        Wxh = nn.Parameter(torch.randn(n_inp, n_hid) * 0.01, requires_grad=True)
        Whh = nn.Parameter(torch.randn(n_hid, n_hid) * 0.01, requires_grad=True)
        Why = nn.Parameter(torch.randn(n_hid, n_out) * 0.01, requires_grad=True)
        bh = nn.Parameter(torch.zeros(n_hid), requires_grad=True)
        by = nn.Parameter(torch.zeros(n_out), requires_grad=True)

        activ = torch.sigmoid

        vhh = nn.Parameter(torch.randn(n_hid, n_hid) * 0.01, requires_grad=True)
        ch = nn.Parameter(torch.zeros(n_hid), requires_grad=True)

    elif init == "tanh-randorth":
        Wxh = torch.nn.Parameter(rand_ortho((n_inp, n_hid), np.sqrt(6./(n_inp + n_hid)), rng), requires_grad=True)
        Whh = torch.nn.Parameter(rand_ortho((n_hid, n_hid), np.sqrt(6./(n_hid + n_hid)), rng), requires_grad=True)
        Why = torch.nn.Parameter(rand_ortho((n_hid, n_out), np.sqrt(6./(n_hid + n_out)), rng), requires_grad=True)
        bh = nn.Parameter(torch.zeros(n_hid), requires_grad=True)
        by = nn.Parameter(torch.zeros(n_out), requires_grad=True)

        activ = torch.tanh

        Vhh = nn.Parameter(rand_ortho((n_hid, n_hid), np.sqrt(6./(n_hid + n_hid)), rng), requires_grad=True)
        ch = nn.Parameter(torch.zeros(n_hid), requires_grad=True)

    # Store the parameters in shared variables
    parameters = {'Wxh': Wxh,
                  'Whh': Whh,
                  'Why': Why,
                  'bh': bh,
                  'by': by,
                  'Vhh': Vhh,
                  'ch': ch}
    
    #########################################
    # TRAINING PHASE                        #
    #########################################

    # The initial state h0 is initialized with 0's
    h0 = torch.zeros(batch_size, n_hid)

    # Define symbolic variables
    x = torch.Tensor() # Assuming tensor3 is a 3D tensor in Theano
    t = torch.Tensor() # Assuming matrix is a 2D tensor in Theano

    i_lr = torch.scalar_tensor(0.0) # assuming scalar is a scalar value in Theano
    f_lr = torch.scalar_tensor(0.0)
    g_lr = torch.scalar_tensor(0.0)

    noise = torch.scalar_tensor(0.0)

    # Define the forward function F(.)
    F = lambda x, hs: activ(torch.mm(hs, Whh) + torch.mm(x, Wxh) + bh)

    # Compute the forward outputs
    h = [h0]
    def forward_function(x_t, h_prev, Whh, Wxh, Why, bh):
        return F(x_t, h_prev)
    
    for x_t in x:
        h.append(F(x_t, h[-1], Whh, Wxh, Why, bh))
    
    # Compute the final output based on the problem type (classificaation or real-valued) and get the global loss
    if task.classifType == 'lastSoftmax':
        # Classification problem - set the last layer to softmax and use cross-entropy loss
        y = torch.nn.functional.softmax(torch.mm(h[-1], Why) + by, dim=1)
        cost = -(t * torch.log(y)).mean(dim=0).sum()
    elif task.classifType == 'lastLinear':
        # Real values output - final step is linear, and the loss is MSE
        y = torch.mm(h[-1], Why) + by
        cost = ((t-y)**2).mean(dim=0).sum()

    # Define the G(.) function
    G = lambda x, hs: activ(torch.mm(x, Wxh) + torch.mm(hs, Vhh) + ch)

    # First target is based on the derivative of the global error w.r.t. the parameters in the final layer
    grad_cost = torch.autograd.grad(cost, h, retain_graph=True)[-1]
    first_target = h[-1] - i_lr * grad_cost

    # Set the local targets for the upstream layers    
    # h_ contains the local targets
    # first_target - deepest hidden layer (e.g. H10)
    # h_[:,0,:][0] - second deepest layer (e.g H9)]
    # ...
    # h_[:,0,:][len(h_)-1] - first hidden layer (e.g. H1)

    