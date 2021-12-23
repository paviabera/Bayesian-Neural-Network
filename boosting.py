# Import relevant packages
import torch
import torch.nn.functional as nnf
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD 
from torch.distributions import constraints
import torchvision as torchv
import torchvision.transforms as torchvt
from torchvision.datasets.mnist import MNIST
from torch import nn
from pyro.infer import SVI, TraceMeanField_ELBO
import pyro
from pyro import poutine
import pyro.optim as pyroopt
import pyro.distributions as dist
import pyro.contrib.bnn as bnn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions.utils import lazy_property
import math
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from pyro.poutine import block, replay, trace
import os
from collections import defaultdict
from functools import partial

import numpy as np
import pyro
import pyro.distributions as dist
import scipy.stats
import torch
import torch.distributions.constraints as constraints
from matplotlib import pyplot
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.poutine import block, replay, trace

class get_loader_data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__ (self):
        return len(self.X_data)

        
# df = pd.read_csv("/home/pranav/BVI_NN/Data Sets/cancer.csv")
# Diagnosis = {'M': 1.0, 'B': 0.0}
# df.diagnosis = [Diagnosis[item] for item in df.diagnosis]
#
# y = df['diagnosis'].values
#
# df = preprocessing.scale(df)
#
# data = torch.tensor(df)
# # Shape of the features and labels
# row = data.shape[0]  # 569
# col = data.shape[1]  # 32
# X = data[0:row, 2:col]
# y = torch.tensor(y)
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#
# train_loader = get_loader_data(torch.FloatTensor(x_train.float()),
#                                 torch.FloatTensor(y_train.float()))
# train_loader = DataLoader(dataset=train_loader, batch_size = 40 ,shuffle=True)
#
#
# test_loader = get_loader_data(torch.FloatTensor(x_test.float()),
#                        torch.FloatTensor(y_test.float()))
# test_loader = DataLoader(dataset=test_loader, shuffle=True)

class Boosting():
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test):
        self.inputSize = x_train.shape[1]
        self.hiddenNodeSize = 128
        self.outputSize = 2
        train_loader = get_loader_data(torch.FloatTensor(x_train.float()),\
                                       torch.FloatTensor(y_train.float()))
        train_loader = DataLoader(dataset=train_loader, batch_size=40, shuffle=True)

        test_loader = get_loader_data(torch.FloatTensor(x_test.float()),
                                      torch.FloatTensor(y_test.float()))
        test_loader = DataLoader(dataset=test_loader, shuffle=True)

        # clear the param store in case we're in a REPL
        pyro.clear_param_store()
        bayesnn = BNN(self.inputSize, self.hiddenNodeSize, self.outputSize)
        bayesnn.infer_parameters(train_loader, num_epochs=50, lr=0.002)


class BNN(nn.Module):
    def __init__(self, inputSize, n_hidden=128, n_classes=2):
        super(BNN, self).__init__()
        self.inputsize = inputSize
        self.n_hidden = n_hidden
        self.n_classes = n_classes
       
    def model(self, X_data, labels=None, kl_factor=1.0):
        #images = images.view(-1, 30)
        #X_data = torch.tensor(X_data)
        print(X_data , "model \n\n\n")
        print(type(X_data) , " model \n\n\n")
        n_X_data = 30 #X_data.size(0)
        # Set-up parameters for the distribution of weights for each layer `a<n>`
        a1_mean = torch.zeros(self.inputsize, self.n_hidden)
        a1_scale = torch.ones(self.inputsize, self.n_hidden) - 0.5
        #z_1 = pyro.sample('z', dist.Normal(a1_mean, a1_scale))
        a1_dropout = torch.tensor(0.25)
        #z_1_scale = torch.zeros(30,self.n_hidden) + 0.1
        
        a2_mean = torch.zeros(self.n_hidden + 1, self.n_hidden) 
        a2_scale = torch.ones(self.n_hidden + 1, self.n_hidden) - 0.5
        #z_2 = pyro.sample('z', dist.Normal(a2_mean, a2_scale))
        a2_dropout = torch.tensor(1.0)
        #z_2_scale = torch.zeros(30,self.n_hidden) + 0.1

        a3_mean = torch.zeros(self.n_hidden + 1, self.n_hidden)
        a3_scale = torch.ones(self.n_hidden + 1, self.n_hidden) - 0.5
        #z_3 = pyro.sample('z', dist.Normal(a3_mean, a3_scale))
        a3_dropout = torch.tensor(1.0)
        #z_3_scale = torch.zeros(30,self.n_hidden) + 0.1

        a4_mean = torch.zeros(self.n_hidden + 1, self.n_classes)
        a4_scale = torch.ones(self.n_hidden + 1, self.n_classes) - 0.5
        #z_4 = pyro.sample('z', dist.Normal(a4_mean, a4_scale))
        #z_4_scale = torch.zeros(30,self.n_hidden) + 0.1

        # Mark batched calculations to be conditionally independent given parameters using `plate`
        with pyro.plate('data', size=n_X_data):
            # Sample first hidden layer
            h1 = pyro.sample('h1', bnn.HiddenLayer(X_data, a1_mean, a1_dropout * a1_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample second hidden layer
            h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_dropout * a2_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample third hidden layer
            h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_dropout * a3_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample output logits
            logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: torch.sigmoid(x,dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))
            
            #print(logits,"SVI model")
            

            # One-hot encode labels
            labels = nnf.one_hot(labels.to(torch.int64)) if labels is not None else None
            
            
            # Condition on observed labels, so it calculates the log-likehood loss when training using VI
            return pyro.sample('lable',dist.OneHotCategorical(logits = logits) , obs = labels)
    
    def guide(self, X_data, index, labels=None, kl_factor=1.0):
        #X_data = torch.tensor(X_data)
        #images = images.view(-1, 30)
        n_X_data = 30 #X_data.size(0)
        # Set-up parameters to be optimized to approximate the true posterior
        # Mean parameters are randomly initialized to small values around 0, and scale parameters
        # are initialized to be 0.1 to be closer to the expected posterior value which we assume is stronger than
        # the prior scale of 1.
        # Scale parameters must be positive, so we constraint them to be larger than some epsilon value (0.01).
        # Variational dropout are initialized as in the prior model, and constrained to be between 0.1 and 1 (so dropout
        # rate is between 0.1 and 0.5) as suggested in the local reparametrization paper
        a1_mean = pyro.param('a1_mean_{}'.format(index), 0.01 * torch.randn(30, self.n_hidden))
        a1_scale = pyro.param('a1_scale_{}'.format(index), 0.1 * torch.ones(30, self.n_hidden),
                              constraint=constraints.greater_than(0.01))
        a1_dropout = pyro.param('a1_dropout_{}'.format(index), torch.tensor(0.25),
                                constraint=constraints.interval(0.1, 1.0))
        a2_mean = pyro.param('a2_mean_{}'.format(index), 0.01 * torch.randn(self.n_hidden + 1, self.n_hidden))
        a2_scale = pyro.param('a2_scale_{}'.format(index), 0.1 * torch.ones(self.n_hidden + 1, self.n_hidden),
                              constraint=constraints.greater_than(0.01)) 
        a2_dropout = pyro.param('a2_dropout_{}'.format(index), torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a3_mean = pyro.param('a3_mean_{}'.format(index), 0.01 * torch.randn(self.n_hidden + 1, self.n_hidden))
        a3_scale = pyro.param('a3_scale_{}'.format(index), 0.1 * torch.ones(self.n_hidden + 1, self.n_hidden),
                              constraint=constraints.greater_than(0.01))
        a3_dropout = pyro.param('a3_dropout_{}'.format(index), torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a4_mean = pyro.param('a4_mean_{}'.format(index), 0.01 * torch.randn(self.n_hidden + 1, self.n_classes))
        a4_scale = pyro.param('a4_scale_{}'.format(index), 0.1 * torch.ones(self.n_hidden + 1, self.n_classes),
                              constraint=constraints.greater_than(0.01))
        # Sample latent values using the variational parameters that are set-up above.
        # Notice how there is no conditioning on labels in the guide!
        with pyro.plate('data', size=n_X_data):
            h1 = pyro.sample('h1', bnn.HiddenLayer(X_data, a1_mean, a1_dropout * a1_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_dropout * a2_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_dropout * a3_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: torch.sigmoid(x),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))
            # print(logits,"SVI guide")
    
    def relbo(model, guide, *args, **kwargs):
        approximation = kwargs.pop('approximation')
        # We first compute the elbo, but record a guide trace for use below.
        traced_guide = trace(guide)
        elbo = pyro.infer.Trace_ELBO(max_plate_nesting=1)
        loss_fn = elbo.differentiable_loss(model, traced_guide, *args, **kwargs)

        # We do not want to update parameters of previously fitted components
        # and thus block all parameters in the approximation apart from z.
        guide_trace = traced_guide.trace
        replayed_approximation = trace(replay(block(approximation, expose=['z']), guide_trace))
        approximation_trace = replayed_approximation.get_trace(*args, **kwargs)

        relbo = -loss_fn - approximation_trace.log_prob_sum()

        # By convention, the negative (R)ELBO is returned.
        return -relbo

    def approximation(X_data, components, weights):
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        result = components[assignment](X_data)
        return result

    def infer_parameters(self, loader, lr=0.002, momentum=0.9, num_epochs=50):
        n_iterations = 2
        initial_approximation = partial(self.guide, index=0)
        components = [initial_approximation]
        weights = torch.tensor([1.])
        wrapped_approximation = partial(self.approximation, components=components, weights=weights)

        for t in range(1, n_iterations + 1):

            # Create guide that only takes data as argument
            wrapped_guide = partial(self.guide, index=t)
            losses = []

            adam_params = {"lr": 0.01, "betas": (0.90, 0.999)}
            optimizer = Adam(adam_params)
            #optimizer = pyroopt.SGD({'lr': lr, 'momentum': momentum, 'nesterov': True}) #TODO: change to Adam(adam_params) to check if we get better accuracy

            # Pass our custom RELBO to SVI as the loss function.
            svi = SVI(self.model, wrapped_guide, optimizer, loss=self.relbo)
            for i in range(num_epochs):
                total_loss = 0.0
                total = 0.0
                correct = 0.0

                for X_data, labels in loader:
                    # Pass the existing approximation to SVI.
                    # print(type(X_data),"main loop")
                    loss = svi.step(X_data, labels, approximation=wrapped_approximation) #TODO: might require to add kl_factor=kl_factor
                    #losses.append(loss)
                    pred = self.forward(X_data, n_samples=1).mean(0)
                    total_loss += loss / len(loader.dataset)
                    total += labels.size(0)

                    correct += (pred.argmax(-1) == labels).sum().item()
                    print(pred)
                    # print(correct)
                    # print(total)
                    param_store = pyro.get_param_store()
                    #if step % 100 == 0:
                        #print('.', end=' ')
                # print(f"[Epoch {i + 1}] loss: {total_loss:.5E} accuracy: {correct / total * 100:.5f}")


            # Update the list of approximation components.
            components.append(wrapped_guide)

            # Set new mixture weight.
            new_weight = 2 / (t + 1)

            # In this specific case, we set the mixture weight of the second component to 0.5.
            if t == 2:
                new_weight = 0.5
            weights = weights * (1 - new_weight)
            weights = torch.cat((weights, torch.tensor([new_weight])))

            # Update the approximation
            wrapped_approximation = partial(self.approximation, components=components, weights=weights)

            # print('Parameters of component {}:'.format(t))
            #scale = pyro.param("scale_{}".format(t)).item()
            #scales.append(scale)
            #loc = pyro.param("loc_{}".format(t)).item()
            #locs.append(loc)
            # print('loc = {}'.format(loc))
            # print('scale = {}'.format(scale))




    def forward(self, X_Data, n_samples=10):
        res = []
        for i in range(n_samples):
            t = poutine.trace(self.guide).get_trace(X_Data)
            res.append(t.nodes['logits']['value'])
        return torch.stack(res, dim=0) 



# total = 0.0
# correct = 0.0
# for images, labels in test_loader:
#     pred = bayesnn.forward(images.view(-1, 30), n_samples=1)
#     #print(labels , "labels" , pred ,"pred")
#     total += labels.size(0)
#     correct += (pred.argmax(-1) == labels).sum().item()
# print(f"Test accuracy: {correct / total * 100:.5f}")