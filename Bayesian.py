# Import relevant packages
import torch
import torch.nn.functional as nnf
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.distributions import constraints
from torch import nn
from pyro.infer import SVI, TraceMeanField_ELBO
import pyro
from pyro import poutine
import pyro.optim as pyroopt
import pyro.distributions as dist
# import pyro.contrib.bnn as bnn
from hidden_layer import HiddenLayer as hnn


import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions.utils import lazy_property
import math
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split



class get_loader_data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class Bayesian():
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test):
        self.inputSize = x_train.shape[1]
        self.hiddenNodeSize = 128
        self.outputSize = 2
        features = x_train.shape[1]
        train_loader = get_loader_data(torch.FloatTensor(x_train.float()),
                                       torch.FloatTensor(y_train.float()))
        train_loader = DataLoader(dataset=train_loader, batch_size=40, shuffle=True)

        test_loader = get_loader_data(torch.FloatTensor(x_test.float()),
                                      torch.FloatTensor(y_test.float()))
        test_loader = DataLoader(dataset=test_loader, shuffle=True)

        pyro.clear_param_store()
        bayesnn = BNN(self.inputSize, self.hiddenNodeSize, self.outputSize)
        bayesnn.infer_parameters(train_loader, num_epochs=50, lr=0.002)

        total = 0.0
        correct = 0.0
        for x_testing, labels in test_loader:
            pred = bayesnn.forward(x_testing.view(-1, features), n_samples=1)
            #print(pred)
            total += labels.size(0)
            correct += (pred.argmax(-1) == labels).sum().item()
        print(f"Test accuracy: {correct / total * 100:.5f}")
        ###TODO
        uncertain_images = []
        for image, _ in test_loader:
            n_samples = 2
            preds = bayesnn.forward(image.view(-1, features), n_samples=n_samples).argmax(-1).squeeze()
            #print(preds)
            pred_sum = [(i, c) for i, c in enumerate(preds.bincount(minlength=10).tolist()) if c > 0]
            if len(pred_sum) > 1:
                uncertain_images.append((image, "\n".join(f"{i}: {c / n_samples:.2f}" for i, c in pred_sum)))
            if len(uncertain_images) >= 64:
                break
        print(uncertain_images)



class BNN(nn.Module):
    def __init__(self, inputsize ,n_hidden, n_classes):
        super(BNN, self).__init__()
        self.inputsize = inputsize
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def model(self, X_data, labels=None, kl_factor=1.0):
        #print("model")
        
        features = X_data.shape[1]
        X_data = X_data.view(-1, features)
        n_X_data = X_data.size(0)
        # Set-up parameters for the distribution of weights for each layer `a<n>`
        a1_mean = torch.zeros(self.inputsize, self.n_hidden)
        a1_scale = torch.ones(self.inputsize, self.n_hidden)
        a1_dropout = torch.tensor(0.25)
        a2_mean = torch.zeros(self.n_hidden + 1, self.n_hidden)
        a2_scale = torch.ones(self.n_hidden + 1, self.n_hidden)
        a2_dropout = torch.tensor(1.0)
        a3_mean = torch.zeros(self.n_hidden + 1, self.n_hidden)
        a3_scale = torch.ones(self.n_hidden + 1, self.n_hidden)
        a3_dropout = torch.tensor(1.0)
        a4_mean = torch.zeros(self.n_hidden + 1, self.n_classes)
        a4_scale = torch.ones(self.n_hidden + 1, self.n_classes)
        # Mark batched calculations to be conditionally independent given parameters using `plate`
        with pyro.plate('data', size=n_X_data):
            # Sample first hidden layer
            h1 = pyro.sample('h1', hnn(X_data, a1_mean, a1_dropout * a1_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample second hidden layer
            h2 = pyro.sample('h2', hnn(h1, a2_mean, a2_dropout * a2_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample third hidden layer
            h3 = pyro.sample('h3', hnn(h2, a3_mean, a3_dropout * a3_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample output logits
            logits = pyro.sample('logits', hnn(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: torch.sigmoid(x,dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))

            #print(logits,"SVI model")


            # One-hot encode labels
            labels = nnf.one_hot(labels.to(torch.int64)) if labels is not None else None
            #print(labels,"lables \n\n")

            # Condition on observed labels, so it calculates the log-likehood loss when training using VI
            return pyro.sample('lable',dist.OneHotCategorical(logits = logits) , obs = labels)

    def guide(self, X_data, labels=None, kl_factor=1.0):
        #print("guide")
        features = X_data.shape[1]
        X_data = X_data.view(-1, features)
        n_X_data = X_data.size(0)
        #print(X_data.shape,"shape of you")
        #print(X_data.shape[1],"shape")
        #print(n_X_data,"N_x_data")
        # Set-up parameters to be optimized to approximate the true posterior
        # Mean parameters are randomly initialized to small values around 0, and scale parameters
        # are initialized to be 0.1 to be closer to the expected posterior value which we assume is stronger than
        # the prior scale of 1.
        # Scale parameters must be positive, so we constraint them to be larger than some epsilon value (0.01).
        # Variational dropout are initialized as in the prior model, and constrained to be between 0.1 and 1 (so dropout
        # rate is between 0.1 and 0.5) as suggested in the local reparametrization paper
        a1_mean = pyro.param('a1_mean', 0.01 * torch.randn(self.inputsize, self.n_hidden))
        a1_scale = pyro.param('a1_scale', 0.1 * torch.ones(self.inputsize, self.n_hidden),
                              constraint=constraints.greater_than(0.01))
        a1_dropout = pyro.param('a1_dropout', torch.tensor(0.25),
                                constraint=constraints.interval(0.1, 1.0))
        a2_mean = pyro.param('a2_mean', 0.01 * torch.randn(self.n_hidden + 1, self.n_hidden))
        a2_scale = pyro.param('a2_scale', 0.1 * torch.ones(self.n_hidden + 1, self.n_hidden),
                              constraint=constraints.greater_than(0.01))
        a2_dropout = pyro.param('a2_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a3_mean = pyro.param('a3_mean', 0.01 * torch.randn(self.n_hidden + 1, self.n_hidden))
        a3_scale = pyro.param('a3_scale', 0.1 * torch.ones(self.n_hidden + 1, self.n_hidden),
                              constraint=constraints.greater_than(0.01))
        a3_dropout = pyro.param('a3_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a4_mean = pyro.param('a4_mean', 0.01 * torch.randn(self.n_hidden + 1, self.n_classes))
        a4_scale = pyro.param('a4_scale', 0.1 * torch.ones(self.n_hidden + 1, self.n_classes),
                              constraint=constraints.greater_than(0.01))
        # Sample latent values using the variational parameters that are set-up above.
        # Notice how there is no conditioning on labels in the guide!
        with pyro.plate('data', size=n_X_data):
            h1 = pyro.sample('h1', hnn(X_data, a1_mean, a1_dropout * a1_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h2 = pyro.sample('h2', hnn(h1, a2_mean, a2_dropout * a2_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h3 = pyro.sample('h3', hnn(h2, a3_mean, a3_dropout * a3_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            logits = pyro.sample('logits', hnn(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: torch.sigmoid(x),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))
            # print(logits,"SVI guide")

    def infer_parameters(self, loader,  num_epochs, lr=0.01, momentum=0.9):
        #print(loader)
        optim = pyroopt.SGD({'lr': lr, 'momentum': momentum, 'nesterov': True})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optim, elbo)
        kl_factor = loader.batch_size / len(loader.dataset)

        for i in range(num_epochs):
            total_loss = 0.0
            total = 0.0
            correct = 0.0

            for X_data, labels in loader:
                #print(X_data.shape)
                #print(X_data.shape,"sjape of data")
                loss = svi.step(X_data, labels, kl_factor=kl_factor)
                pred = self.forward(X_data, n_samples=1).mean(0)
                #print(pred)
                total_loss += loss / len(loader.dataset)
                total += labels.size(0)

                correct += (pred.argmax(-1) == labels).sum().item()
                #print(correct)
                #print(total)
                param_store = pyro.get_param_store()
            print(f"[Epoch {i + 1}] loss: {total_loss:.5E} accuracy: {correct / total * 100:.5f}")

    def forward(self, X_data, n_samples=10):
        res = []
        for i in range(n_samples):
            t = poutine.trace(self.guide).get_trace(X_data)
            res.append(t.nodes['logits']['value'])
        return torch.stack(res, dim=0)
