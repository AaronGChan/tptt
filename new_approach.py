from rnn_dtp_v2 import TPTT_RNN, sample_length
import numpy as np
import sys
import argparse
from collections import OrderedDict
from torch.autograd import Variable

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
    if init == "sigmoid":
        activ = torch.sigmoid
        rnn = TPTT_RNN(n_inp=n_inp, n_hid=n_hid, n_out=n_out, init=init, activ=activ, task=task, ilr=i_learning_rate, flr=f_learning_rate, glr=g_learning_rate)
    elif init == "tanh-randorth":
        activ = torch.tanh
        rnn = TPTT_RNN(n_inp=n_inp, n_hid=n_hid, n_out=n_out, init=init, activ=activ, task=task, ilr=i_learning_rate, flr=f_learning_rate, glr=g_learning_rate)
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
    best_score = 0
    
    while (training) and (n <= maxiter):
        train_x, train_y = task.generate(batch_size,
                                         sample_length(min_length,
                                                       max_length, rng))
        train_x = Variable(torch.from_numpy(train_x))
        train_y = Variable(torch.from_numpy(train_y))
        # step g
        logits, h = rnn.forward(train_x)
        h_target = rnn.target_propagation(train_x, logits, h, train_y)
        tr_cost, f_Whh, f_Wxh, f_Why, f_by, f_bh = rnn.update_params(train_x, train_y, h_target, i_learning_rate, f_learning_rate, g_learning_rate)
        # Update the accumulation variables
        avg_cost += tr_cost

        avg_dWhh_norm += f_Whh
        avg_dWxh_norm += f_Wxh
        avg_dWhy_norm += f_Why
        avg_dbh_norm += f_bh
        avg_dhy_norm += f_by
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
                _cost, _error = rnn.eval_step(valid_x, valid_y)                
                error = error + _error
                valid_cost = valid_cost + _cost
 
            # Compute the average error and cost
            error = error*100. / float(val_size // val_batch)
            valid_cost = valid_cost / float(val_size // val_batch)

            # Get the spectral radius of the Whh and Vhh matrices            
            rho_Whh =np.max(abs(np.linalg.eigvals(rnn.Whh.get_value())))
            rho_Vhh =np.max(abs(np.linalg.eigvals(rnn.Vhh.get_value())))

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
        task = TempOrderTask(rng, "float32")
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
    fit(rng, i_learning_rate, f_learning_rate, g_learning_rate, n_hid, init, \
        batch_size, max_length, min_length, task, maxiter, chk_interval, noise, \
        val_size, val_batch, gd_opt, args.task, wxh_updates)
if __name__=='__main__':

    main(sys.argv)