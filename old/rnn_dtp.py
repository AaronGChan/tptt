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
    A = -irange + (2 * torch.rand(shape) - 1)
    U, _, V = torch.svd(A)
    return torch.mm(U, torch.mm(torch.eye(U.shape[1], V.shape[0]), V))

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

    # Define symbolic variables, inital state?
    # 10, 20, 6
    # seq, batch_size, 
    x = torch.zeros(10, 20, 6)#torch.Tensor() # Assuming tensor3 is a 3D tensor in Theano 
    t = torch.zeros(20, 4)#torch.Tensor() # Assuming matrix is a 2D tensor in Theano
    #breakpoint()
    i_lr = torch.scalar_tensor(0.0) # assuming scalar is a scalar value in Theano
    f_lr = torch.scalar_tensor(0.0)
    g_lr = torch.scalar_tensor(0.0)

    noise = torch.scalar_tensor(0.0)

    # Define the forward function F(.)
    #F = lambda x, hs: activ(torch.mm(hs, Whh) + torch.mm(x, Wxh) + bh)
    def F(x, hs):
        breakpoint()
        return activ(torch.mm(hs, Whh) + torch.mm(x, Wxh) + bh)
    # Compute the hidden activations (h_t) in forward pass
    h = [h0]
    def forward_function(x_t, h_prev, Whh, Wxh, Why, bh):
        return F(x_t, h_prev)
    
    for x_t in x:
        h.append(F(x_t, h[-1]))
    
    # Compute the final output based on the problem type (classificaation or real-valued) and get the global loss
    if task.classifType == 'lastSoftmax':
        # Classification problem - set the last layer to softmax and use cross-entropy loss
        y = torch.nn.functional.softmax(torch.mm(h[-1], Why) + by, dim=1)
        return -(t * torch.log(y)).mean(dim=0).sum()
    elif task.classifType == 'lastLinear':
        # Real values output - final step is linear, and the loss is MSE
        y = torch.mm(h[-1], Why) + by
        cost = ((t-y)**2).mean(dim=0).sum()

    # Define the G(.) function
    G = lambda x, hs: activ(torch.mm(x, Wxh) + torch.mm(hs, Vhh) + ch)

    # First target is based on the derivative of the global error w.r.t. the parameters in the final layer
    grad_cost = torch.autograd.grad(cost(t, y), h, retain_graph=True)[-1]
    first_target = h[-1] - i_lr * grad_cost

    # Set the local targets for the upstream layers    
    # h_ contains the local targets
    # first_target - deepest hidden layer (e.g. H10)
    # h_[:,0,:][0] - second deepest layer (e.g H9)]
    # ...
    # h_[:,0,:][len(h_)-1] - first hidden layer (e.g. H1)
    
    # Compute h_ using torch.scan equivalent
    h_ = []
    h_hat_t = None

    # implement equation 6 from the paper
    # we begin from the last time step of the horizontal hidden layer and obtain corresponding h_hat_t at every time step until t=1
    for index in reversed(range(len(x))):
        if index == len(x)-1:
            h_hat_t = first_target
        else:
            h_hat_t = h[index] - G(x[index+1], h[index+1]) + G(x[index+1], h_hat_t)
        print(f"index {index}: shape of h_hat_t -> {h_hat_t.shape}")
        h_.append(h_hat_t)

    # Merge first_target and h_ and get an unified tensor with all targets
    # first_target = torch.reshape(first_target, [1, first_target.shape[0], first_target.shape[1]])
    # h_ = torch.cat([first_target, h_])

    # Reverse the order f h_ to get [H_0, H_1, H_2 ....]
    h_ = torch.flip(h_, dims=[0]) # h_[::-1]

    # gradients of feedback (inverse) mapping
    
    # splice h0 and h in h_offset, and remove H for the last layer (we don't need it)

    h_offset = torch.cat([torch.reshape(h0, [1, h0.shape[0], h0.shape[1]]), h])[:-1, :, :]

    # Add gaussian noise
    h_offset_c = h_offset + gaussian(h_offset.shape, noise)

    # Loop over h_offset & x so that torch.autograd.grad(mse(G(x[t], F(x[t], h[t-1])), h[t-1]), [Vhh, Ch], consider_constant=[x[t], F(x[t], h[t-1]), h[t-1]])

    # initialize the gradients
    dVhh_sum = torch.zeros_like(Vhh)
    dCh_sum = torch.zeros_like(ch)

    for x_t, h_tm1 in zip(x, h_offset_c):
        # calculate G(x_t, F(x_t, h_tm1))
        output = G(x_t, F(x_t, h_tm1))

        # calculate gradients
        mse_loss = mse(output, h_tm1)

        # Detach constants from computational graph
        output = output.detach()
        h_tm1 = h_tm1.detach()

        dVhh, dCh = torch.autograd.grad(mse_loss, [Vhh, ch])

        dVhh_sum += dVhh
        dCh_sum += dCh

        # Reset gradients for the next iteration
        Vhh.grad.zero_()
        ch.grad.zero_()
    
    # Remove accumulated gradients
    dVhh = dVhh_sum
    dCh = dCh_sum
    del dVhh_sum
    del dCh_sum

    # Compute the norm of the updates of G(.)
    g_norm_theta = torch.sqrt(torch.sum(dVhh**2) + torch.sum(dCh**2))

    # Graduents of the feedforward
    if (wxh_updates == "bptt"):
        # using TPTT for the Whh and bh updates
        def compute_grad(x_t, h_t, h_tm1, h_hat_t, Wxh, Whh, bh):
            x_t_detached = x_t.detach() # Detach x_t
            h_tm1_detached = h_tm1.detach() # Detach h_tm1
            h_hat_t_detached = h_hat_t.detach() # Detach h_hat_t
            return torch.autograd.grad(mse(F(x_t, h_tm1), h_hat_t), [Whh, bh], retain_graph=True)
        
        grad_output = [compute_grad(x_t, h_t, h_tm1, h_hat_t, Wxh, Whh, bh) for x_t, h_t, h_tm1, h_hat_t in zip(x, h, h_offset, h_)]

        dWhh, dbh = zip(*grad_output)
        dWhh = torch.stack(dWhh).sum(dim=0)
        dbh = torch.stack(dbh).sum(dim=0)

        # Using BPTT for the Wxh, Why, and by updates
        dWxh, dWhy, dby = torch.autograd.grad(cost, [Wxh, Why, by])

    else:
        # Using TPTT for the Whh, bh, and Wxh updates
        def compute_grad(x_t, h_t, h_tm1, h_hat_t, Wxh, Whh, bh):
            x_t_detached = x_t.detach()  # Detach x_t
            h_tm1_detached = h_tm1.detach()  # Detach h_tm1
            h_hat_t_detached = h_hat_t.detach()  # Detach h_hat_t
            return torch.autograd.grad(mse(F(x_t_detached, h_tm1_detached), h_hat_t_detached), [Whh, bh, Wxh], retain_graph=True)

        grad_output = [compute_grad(x_t, h_t, h_tm1, h_hat_t, Wxh, Whh, bh) for x_t, h_t, h_tm1, h_hat_t in zip(x, h, h_offset, h_)]

        dWhh, dbh, dWxh = zip(*grad_output)
        dWhh = torch.stack(dWhh).sum(dim=0)
        dbh = torch.stack(dbh).sum(dim=0)
        dWxh = torch.stack(dWxh).sum(dim=0)

        # Compute dWhy and dby using BPTT
        dWhy, dby = torch.autograd.grad(cost, [Why, by])

    # Add up the dWhh and dbh corrections
    dWhh = dWhh.sum(dim=0)
    dbh = dbh.sum(dim=0)

    # Set the optimisation technique
    if gd_opt == "vanilla":
        # Vanilla SGD
        updates_g = vanilla_sgd([Vhh, ch], [dVhh, dCh], g_lr)
        updates_f = vanilla_sgd([Wxh, Whh, bh, Why, by], [dWxh, dWhh, dbh, dWhy, dby], f_lr)
    
    elif gd_opt == "nesterov":
        # Nesterov accelerated gradient
        updates_g = nesterov_momentum([Vhh, ch], [dVhh, dCh], g_lr)
        updates_f = nesterov_momentum([Wxh, Whh, bh, Why, by], [dWxh, dWhh, dbh, dWhy, dby], f_lr)

    # Compute the norm of each feedforward update matrix
    dWhh_norm = torch.sqrt(torch.sum(dWhh**2))
    dWxh_norm = torch.sqrt(torch.sum(dWxh**2))
    dWhy_norm = torch.sqrt(torch.sum(dWhy**2))
    dby_norm = torch.sqrt(torch.sum(dby**2))
    dbh_norm = torch.sqrt(torch.sum(dbh**2))

    # Define a forward step function
    def f_step(x, y, i_lr, f_lr):
        return cost(x, y), dWhh_norm, dWxh_norm, dWhy_norm, dby_norm, dbh_norm
    
    # Define a feedback step function 
    def g_step(x, g_lr, noise):
        return g_norm_theta
    
    #########################################
    # VALIDATION PHASE                      #
    #########################################

    # Define symbolic variables for the validation phase
    h0_val = torch.zeros(val_batch, n_hid)

    x_val = torch.Tensor()
    t_val = torch.Tensor()

    # Define forward pass function
    def forward_pass(x_t, h_prev, Whh, Wxh, Why):
        return activ(torch.matmul(h_prev, Whh) + torch.matmul(x_t, Wxh) + bh)

    # Set a forward pass
    h_val = [h0_val]
    h_prev = h0_val
    for x_t in x_val:
        h_prev = forward_pass(x_t, h_prev, Whh, Wxh, Why, bh)
        h_val.append(h_prev)

    h_val = torch.stack(h_val)

    # Define a step function for the validation pass
    def eval_step(x_val, t_val): # h_val, Why, by, task
        # Compute the final output based on the problem type (classification or real-value), get the global loss, and measure the prediction error
        if task.classifType == "lastSoftmax":
            # classification problem - set the last layer to softmax and use cross-entropy loss
            y_val = nn.softmax(torch.matmul(h_val[-1], Why) + by, dim=1)
            cost_val = -(t_val * torch.log(y_val)).mean(axis=0).sum()
            error_val = (torch.argmax(y_val, dim=1) != torch.argmax(t_val, dim=1)).float().mean()
        elif task.classifType == "lastLinear":
            # Real-values output - final step is linear, and the loss is MSE
            y_val = torch.matmul(h_val[-1], Why) + by
            cost_val = ((t_val - y_val)**2).mean(axis=0).sum()
            # An example in the mini-batch is considered successfully predicted if 
            # the error between the prediction and the target is below 0.04
            error_val = (((t_val - y_val)**2).sum(axis=1) > 0.04).float().mean()

        return cost_val, error_val
    
    print("******************************************************")
    print("Training starts...")
    print("******************************************************")
    
    # Control variable for the tarining loop
    training = True
    
    # Iteration number
    n = 1
    
    # Cost accumulator variable
    avg_cost = 0
    
    # Gradient norm accumulator variables
    avg_dWhh_norm = 0
    avg_dWxh_norm = 0
    avg_dWhy_norm = 0
    avg_dbh_norm = 0
    avg_dhy_norm = 0
    avg_g_norm = 0
    
    patience = 300
    
    # Measure the initial accuracy
    valid_x, valid_y = task.generate(val_batch, 
                                     sample_length(min_length, 
                                                   max_length, rng))
    best_score = eval_step(valid_x, valid_y) [1] * 100

    # Repeat until convergence or upon reaching the maxiter limit
    while (training) and (n <= maxiter):

        # Get a mini-batch of training data
        train_x, train_y = task.generate(batch_size,
                                         sample_length(min_length,
                                                       max_length, rng))

        # Perform a feedback step (set targets)
        g_norm = g_step(train_x, g_learning_rate, gaussian_noise)
        
        # Perform a forward step
        tr_cost, f_Whh, f_Wxh, f_Why, f_by, f_bh = f_step(train_x, train_y, i_learning_rate, f_learning_rate)

        # Update the accumulation variables
        avg_cost += tr_cost
        avg_dWhh_norm += f_Whh
        avg_dWxh_norm += f_Wxh
        avg_dWhy_norm += f_Why
        avg_dbh_norm += f_bh
        avg_dhy_norm += f_by
        avg_g_norm += g_norm

        if (n % chk_interval == 0):
            patience -= 1
            # Time to check the performance on the validation set

            # If the cost is NaN, abort the training
            avg_cost = avg_cost / float(chk_interval)

            if not torch.isfinite(tr_cost):
                print("******************************************************")
                print("Cost is NAN. Training aborted. Best error : %07.3f%%" % best_score)
                print("******************************************************")
                print("------------------------------------------------------")
                return (n-1),best_score
            
            # Get the average of the accumulation variables

            avg_g_norm = avg_g_norm / float(chk_interval)                    

            avg_dWhh_norm = avg_dWhh_norm/ float(chk_interval)                    
            avg_dWxh_norm = avg_dWxh_norm/ float(chk_interval)                    
            avg_dWhy_norm = avg_dWhy_norm/ float(chk_interval)                    
            avg_dbh_norm = avg_dbh_norm/ float(chk_interval)                    
            avg_dhy_norm = avg_dhy_norm/ float(chk_interval)                    

            # Accumulation variables for the validation cost and error
            valid_cost = 0
            error = 0

            # Get the number of mini-batches needed to cover the desired
            # validation sample and loop over them            
            for dx in range(val_size // val_batch):
                    
                # Get a mini-batch for validation
                valid_x, valid_y = task.generate(val_batch, 
                                                 sample_length(min_length, 
                                                               max_length, rng))

                # Take a validation step and get the cost and error from
                # this mini-batch
                _cost, _error = eval_step(valid_x, valid_y)                
                error = error + _error
                valid_cost = valid_cost + _cost
 
            # Compute the average error and cost
            error = error*100. / float(val_size // val_batch)
            valid_cost = valid_cost / float(val_size // val_batch)

            # Get the spectral radius of the Whh and Vhh matrices            
            rho_Whh =np.max(abs(np.linalg.eigvals(Whh.get_value())))
            rho_Vhh =np.max(abs(np.linalg.eigvals(Vhh.get_value())))

            if (rho_Whh>20 or rho_Vhh>20):
                print("Rho exploding. Aborting....")
                training = False

            # Is the new error lower than our best? Update the best
            if error < best_score:
                patience = 300
                best_score = error
                    
            if (patience <= 0):
                print("No improvement over 30'000 samples. Aborting...")
                training = False
                
            # Print the results from the validation
            print("Iter %07d" % n, ":",
                  "cost %05.3f, " % avg_cost,
                  "|Whh| %7.5f, " % avg_dWhh_norm,
                  "r %01.3f," % rho_Whh,
                  "|bh| %7.3f, " % avg_dbh_norm,
                  "|Wxh| %7.3f, " % avg_dWxh_norm,
                  "|Why| %7.3f, " % avg_dWhy_norm,
                  "|by| %7.3f, " % avg_dhy_norm,
                  "|g| %7.5f, " % avg_g_norm,
                  "r %01.3f," % rho_Vhh,
                  "err %07.3f%%, " % error,
                  "best err %07.3f%%" % best_score)

            # Is the error below 0.0001? If yes, the problem has been solved
            if error < .0001 and np.isfinite(valid_cost):
                training = False
                print("PROBLEM SOLVED!")


            # Reset the accumulators
            avg_cost = 0
            avg_dWhh_norm = 0
            avg_dWxh_norm = 0
            avg_dWhy_norm = 0
            avg_dbh_norm = 0
            avg_dhy_norm = 0
            
            avg_g_norm = 0

        # Increase the iteration counter
        n += 1    
        
    # Training completed. Print the final validation error.
    print("******************************************************")
    print("Training completed. Final best error : %07.3f%%" % best_score)
    print("******************************************************")
    print("------------------------------------------------------")

    return (n-1),best_score

def main(args): 

    # Set a random seed for reproducibility
    rng = np.random.RandomState(1234)

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Runs a TPTT RNN test against the pathological\
                                     tasks defined in Hochreiter, S. and Schmidhuber, J. \
                                     (1997). Long short-term memory. Neural Computation, \
                                     9(8), 1735–1780\nThis work is licensed under the \
                                     Creative Commons Attribution 4.0 International License.")

    parser.add_argument("--task", help="Pathological task", choices=["temporal", "temporal3", "addition", "perm"],
                        required=True)

    parser.add_argument("--maxiter",help="Maximum number of iterations", 
                        default = 100000, required=False, type=int)
    
    parser.add_argument("--batchsize",help="Size of the minibatch", 
                        default = 20, required=False, type=int)

    parser.add_argument("--min",help="Minimal length of the task", 
                        default = 10, required=False, type=int)

    parser.add_argument("--max",help="Maximal length of the task", 
                        default = 10, required=False, type=int)

    parser.add_argument("--chk",help="Check interval", 
                        default = 100, required=False, type=int)

    parser.add_argument("--hidden",help="Number of units in the hidden layer", 
                        default = 100, required=False, type=int)

    parser.add_argument("--opt", help="Optimizer", choices=["vanilla", "nesterov"],
                        default = "nesterov", required=False)

    parser.add_argument("--init", help="Weight initialization and activation function", choices=["tanh-randorth", "sigmoid"],
                        default = "tanh-randorth", required=False)

    parser.add_argument("--ilr",help="Initial learning rate", default = 0.1, 
                        required=False, type=float)

    parser.add_argument("--flr",help="Forward learning rate", default = 0.01, 
                        required=False, type=float)

    parser.add_argument("--glr",help="Feedback learning rate", default = 0.001, 
                        required=False, type=float)
    
    parser.add_argument("--wxh_updates", help="Update mechanism for Wxh", choices=["bptt", "tptt"],
                        default = "tptt", required=False)

    parser.add_argument("--noise", help="Injected Gaussian noise", 
                        default = 0.0, required=False, type=float)

    args = parser.parse_args()

    # Maximal length of the task and minimal length of the task.
    # If you want to run an experiment were sequences have fixed length, set
    # these to hyper-parameters to the same value. Otherwise each batch will
    # have a length randomly sampled from [min_length, max_length]
    min_length = args.min
    max_length = args.max
    
    noise = args.noise
    
    # Get the problem type and instantiate the respective generator
    if args.task == "temporal":
        task = TempOrderTask(rng, torch.float32)
    if args.task == "temporal3":
        task = TempOrder3bitTask(rng, torch.float32)
    elif args.task == "addition":
        task = AddTask(rng, torch.float32)
    elif args.task == "perm":
        task = PermTask(rng, torch.float32)
    
    # Set the maximum number of iterations
    maxiter = args.maxiter
    
    # Update mechanism for Wxh
    wxh_updates = args.wxh_updates

    # Set the mini-batch size
    batch_size = args.batchsize

    # Set the number of iterations between each validation
    chk_interval = args.chk

    # Set the number of neurons in the hidden layer
    n_hid = args.hidden

    # Set the random weights initialisation and the optimisation techniuqe
    init   = args.init
    gd_opt = args.opt
    
    # Set the size and number of mini-batches for the validation phase
    val_size  = 10000
    val_batch = 1000
    
    # Set the learning rates and Gaussian noise decay iteration
    i_learning_rate = args.ilr
    f_learning_rate = args.flr
    g_learning_rate = args.glr
    
    # Train the network
    fit(rng, i_learning_rate, f_learning_rate, g_learning_rate, n_hid, init, \
        batch_size, max_length, min_length, task, maxiter, chk_interval, noise, \
        val_size, val_batch, gd_opt, args.task, wxh_updates)

if __name__=='__main__':

    main(sys.argv)