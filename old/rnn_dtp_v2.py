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


class TPTT_RNN(nn.Module):
    """
    class for RNN as designed for TPTT

    :param nn: pytorch neural network
    :type nn: pytorch nn module
    """

    def __init__(self, n_inp, n_hid, n_out, init, activ, task, ilr, flr, glr):
        """
        constructor

        :param n_inp: input size
        :type n_inp: int
        :param n_hid: hidden layer size
        :type n_hid: int
        :param n_out: output layer size
        :type n_out: int
        :param init: hidden unit initialization/activation
        :type init: str
        :param activ: activation function
        :type activ: lambda
        :param task: task information (addition, temporal order)
        :type task: str 
        """
        super(TPTT_RNN, self).__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.init = init
        self.activ = activ
        self.task = task
        self.i_lr = ilr
        self.f_lr = flr
        self.g_lr = glr

        if init == "sigmoid":
            self.Wxh = nn.Parameter(torch.randn(n_inp, n_hid) * 0.01)
            self.Whh = nn.Parameter(torch.randn(n_hid, n_hid) * 0.01)
            self.Why = nn.Parameter(torch.randn(n_hid, n_out) * 0.01)
            self.bh = nn.Parameter(torch.zeros(n_hid))
            self.by = nn.Parameter(torch.zeros(n_out))

            self.activ = torch.sigmoid

            self.Vhh = nn.Parameter(torch.randn(n_hid, n_hid) * 0.01)
            self.ch = nn.Parameter(torch.zeros(n_hid))

        elif init == "tanh-randorth":
            self.Wxh = nn.Parameter(rand_ortho((n_hid, n_inp), np.sqrt(6./(n_inp + n_hid))).T)
            self.Whh = nn.Parameter(rand_ortho((n_hid, n_hid), np.sqrt(6./(n_hid + n_hid))))
            self.Why = nn.Parameter(rand_ortho((n_hid, n_out), np.sqrt(6./(n_hid + n_out))))
            self.bh = nn.Parameter(torch.zeros(n_hid))
            self.by = nn.Parameter(torch.zeros(n_out))

            self.activ = torch.tanh

            self.Vhh = nn.Parameter(rand_ortho((n_hid, n_hid), np.sqrt(6./(n_hid + n_hid))))
            self.ch = nn.Parameter(torch.zeros(n_hid))
    def _f(self, x, hs):
            return self.activ(torch.matmul(hs, self.Whh) + torch.matmul(x, self.Wxh) + self.bh)
    def forward(self, x, h0=None):
        """
        forward pass for TPTT

        :param x: input
        :type x: [:, :, :]
        :param h0: initial hidden state, defaults to None
        :type h0: float, optional
        :return: logits of the network (no softmax)
        :rtype: [:, :]
        """
        seq_len, batch_size, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(batch_size, self.n_hid, device=x.device)
        # breakpoint()
        h = torch.empty(seq_len, batch_size, self.n_hid, requires_grad=True)
        h[0, :, :] = self._f(x[0, :, :], h0)
        # ht = h0
        h_stack = [h[0, :, :]]
        for t in range(1, seq_len):
            h[t, :, :] = self._f(x[t, :, :], h[t - 1])
            h_t = h[t, :, :]
            h_stack.append(h_t)
        
        h_stack_stack = torch.stack(h_stack, dim=0)
        h = h_stack_stack
        logits = h[-1, :, :] @ self.Why + self.by#torch.matmul(h[-1, :, :].clone(), self.Why) + self.by
        breakpoint()
        logits.requires_grad=True
        return logits, h

    def target_propagation(self, x, y_pred, h, y_true):
        """
        applying the linearly corrected formula given in TPTT to get h_hat_t

        :param x: input
        :type x: [batch, seq, input]
        :param y_pred: predicted logits
        :type y_pred: [:, :, :]
        :param h: hidden state
        :type h: [:, :, :]
        :param y_true: expected output
        :type y_true: [:, :]
        :return: h_hat_t/h_targets
        :rtype: [:,:,:]
        """
        seq_len, batch_size, _ = x.size()
        h0 = torch.zeros(batch_size, self.n_hid, device=x.device)
        first_target = h[-1, :, :] - self.i_lr * torch.autograd.grad(self.cost(y_pred, y_true), h, retain_graph=True)[0][-1, :, :]
        #breakpoint()
        h_targets = [first_target]
        for t in range(seq_len-1, -1, -1):
            xt = x[t, :, :]
            ht = h[t, :, :]
            ht_target = ht - self.G(xt, ht) + self.G(xt, h_targets[-1])
            h_targets.append(ht_target)

        h_targets = h_targets[::-1]
        return h_targets
    
    def G(self, x, h):
        """
        G function

        :param x: input
        :type x: [:, :, :]
        :param h: hidden state
        :type h: [:, :, :]
        :return: previous hidden state in time
        :rtype: [:, :, :]
        """
        return self.activ(torch.matmul(x, self.Wxh) + torch.matmul(h, self.Vhh) + self.ch)
    #@staticmethod
    def cost(self, y, t):
        """
        cost of the output

        :param y: predicted output
        :type y: [:, :]
        :param t: expected output
        :type t: [:, :]
        :return: cost/loss value
        :rtype: [:, :]
        """
        breakpoint()
        if self.task.classifType == 'lastSoftmax':
            # torch.nn.functional.cross_entropy() takes logits as input and applies softmax to them internally
            return F.cross_entropy(y, t.argmax(dim=1))
        elif self.task.classifType == "lastLinear":
            return F.mse_loss(y, t)

    def update_params(self, x, y, h_targets, i_lr, f_lr, g_lr):
        output, h = self.forward(x)
        cost = self.cost(y, output)
        cost.requires_grad = True
        breakpoint()
        dWhy, dby = torch.autograd.grad(cost, (self.Why, self.by), retain_graph=True)
        dWhh, dbh, dWxh = [], [], []

        for t in range(x.size(0)):
            xt = x[:, t, :]
            ht = h[:, t, :]
            ht_target = h_targets[t]
            grads = torch.autograd.grad(F.mse_loss(self.activ(torch.matmul(ht, self.Whh) + torch.matmul(xt, self.Wxh) + self.bh), ht_target),
            [self.Whh, self.bh, self.Wxh],
            retain_graph=True)
            dWhh.append(grads[0])
            dbh.append(grads[1])
            dWxh.append(grads[2])
        
        dWhh = torch.stack(dWhh, dim=0).sum(dim=0)
        dbh = torch.stack(dbh, dim=0).sum(dim=0)
        dWxh = torch.stack(dWxh, dim=0).sum(dim=0)

        dVhh, dch = torch.autograd.grad(self.feedback_cost(x, h, h_targets), [self.Vhh, self.ch])

        self.Why.data -= f_lr * dWhy
        self.by.data -= f_lr * dby
        self.Whh.data -= f_lr * dWhh
        self.bh.data -= f_lr * dbh
        self.Wxh.data -= f_lr * dWxh
        self.Vhh.data -= g_lr * dVhh
        self.ch.data -= g_lr * dch
        return cost, self.Whh.data, self.Wxh.data, self.by.data, self.bh.data
    
    def feedback_cost(self, x, h, h_targets):
        cost = 0
        for t in range(x.size(1)):
            xt = x[:, t, :]
            ht = h[:, t, :]
            ht_target = h_targets[t]
            cost += F.mse_loss(self.G(xt, ht), ht_target)
        return cost
    

    def eval_step(self, x_val, t_val):
        y_val, h_val = self.forward(x_val)
        if self.task.classifType == "lastSoftmax":
            y_val = nn.softmax(torch.matmul(h_val[-1], self.Why) + self.by, dim=1)
            cost_val = -(t_val * torch.log(y_val)).mean(axis=0).sum()
            error_val = (torch.argmax(y_val, dim=1) != torch.argmax(t_val, dim=1)).float().mean()
        elif self.task.classifType == "lastLinear":
            # Real-values output - final step is linear, and the loss is MSE
            y_val = torch.matmul(h_val[-1], self.Why) + self.by
            cost_val = ((t_val - y_val)**2).mean(axis=0).sum()
            # An example in the mini-batch is considered successfully predicted if 
            # the error between the prediction and the target is below 0.04
            error_val = (((t_val - y_val)**2).sum(axis=1) > 0.04).float().mean()
        return cost_val, error_val
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

def rand_ortho(shape, irange):
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
