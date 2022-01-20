# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        We recommend setting the lrate to 0.01 for part 1

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.net = nn.Sequential(nn.Linear(in_size, 128), nn.ReLU(), nn.Linear(128, out_size))
        self.optimizer = optim.SGD(self.net.parameters(), lr=lrate,weight_decay=0.01)

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
        return self.net(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optimizer.zero_grad()
        yhat = self.forward(x.requires_grad_(True))
        print(y.size())
        L = self.loss_fn(yhat,y)
        L.backward()
        self.optimizer.step()
        return L


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """
    losses = []
    net = NeuralNet(lrate=0.09, loss_fn=nn.CrossEntropyLoss(), in_size=3072, out_size=2)
    train_set = (train_set - torch.mean(train_set,dim=0)) / torch.std(train_set,dim=0)
    N = len(train_labels)
    for n in range(n_iter):
        start = n * batch_size % N
        end = (n + 1) * batch_size % N
        losses.append(net.step(train_set[start:end, :], train_labels[start:end]))
    yhats = np.zeros(len(dev_set))

    dev_set = (dev_set - torch.mean(dev_set,dim=0)) / torch.std(dev_set,dim=0)
    outputs = net(dev_set)
    for i in range(len(outputs)):
        out = outputs[i]
        yhat = out.max(dim=0)[1]
        yhats[i] = yhat

    return losses, yhats, net
