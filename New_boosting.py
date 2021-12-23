import os
from collections import defaultdict
from functools import partial
#import pyro.contrib.bnn as bnn
import torch.nn.functional as nnf
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
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from hidden_layer import HiddenLayer as hnn

smoke_test = ('CI' in os.environ)

# this is for running the notebook in our testing framework
n_steps = 2 if smoke_test else 12000
pyro.set_rng_seed(2)

# clear the param store in case we're in a REPL
pyro.clear_param_store()


class get_loader_data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__ (self):
        return len(self.X_data)
# Sample observations from a Normal distribution with loc 4 and scale 0.1
df = pd.read_csv("Data Sets/cancer.csv")
Diagnosis = {'M': 1.0, 'B': 0.0} 
df.diagnosis = [Diagnosis[item] for item in df.diagnosis]

y = df['diagnosis'].values

df = preprocessing.scale(df)

data_1 = torch.tensor(df)
# Shape of the features and labels
row = data_1.shape[0]  # 569
col = data_1.shape[1]  # 32
X = data_1[0:row, 2:col]
y = torch.tensor(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

train_loader = get_loader_data(torch.FloatTensor(x_train.float()), 
                                torch.FloatTensor(y_train.float()))
train_loader = DataLoader(dataset=train_loader, batch_size = 40 ,shuffle=True)


test_loader = get_loader_data(torch.FloatTensor(x_test.float()), 
                       torch.FloatTensor(y_test.float()))
test_loader = DataLoader(dataset=test_loader, shuffle=True)


def guide(X_data, index, kl_factor=1.0 ):
    #print("guide")
    #images = images.view(-1, 30)
    #print(X_data.shape)
    features = X_data.shape[1]
    X_data = X_data.view(-1, features)
    n_X_data = X_data.size(1)
    #print(n_X_data,"n_x_data")
    # Set-up parameters to be optimized to approximate the true posterior
    # Mean parameters are randomly initialized to small values around 0, and scale parameters
    # are initialized to be 0.1 to be closer to the expected posterior value which we assume is stronger than
    # the prior scale of 1.
    # Scale parameters must be positive, so we constraint them to be larger than some epsilon value (0.01).
    # Variational dropout are initialized as in the prior model, and constrained to be between 0.1 and 1 (so dropout
    # rate is between 0.1 and 0.5) as suggested in the local reparametrization paper
    a1_mean = pyro.param('a1_mean_{}'.format(index), 0.01 * torch.randn(n_X_data, 128))

    a1_scale = pyro.param('a1_scale_{}'.format(index), 0.1 * torch.ones(n_X_data, 128),
                           constraint=constraints.greater_than(0.01))
    a1_dropout = pyro.param('a1_dropout_{}'.format(index), torch.tensor(0.25),
                           constraint=constraints.interval(0.1, 1.0))
    a2_mean = pyro.param('a2_mean_{}'.format(index), 0.01 * torch.randn(128 + 1, 128))
    a2_scale = pyro.param('a2_scale_{}'.format(index), 0.1 * torch.ones(128 + 1, 128),
                           constraint=constraints.greater_than(0.01)) 
    a2_dropout = pyro.param('a2_dropout_{}'.format(index), torch.tensor(1.0),
                           constraint=constraints.interval(0.1, 1.0))
    a3_mean = pyro.param('a3_mean_{}'.format(index), 0.01 * torch.randn(128 + 1, 128))
    a3_scale = pyro.param('a3_scale_{}'.format(index), 0.1 * torch.ones(128 + 1, 128),
                           constraint=constraints.greater_than(0.01))
    a3_dropout = pyro.param('a3_dropout_{}'.format(index), torch.tensor(1.0),
                           constraint=constraints.interval(0.1, 1.0))
    a4_mean = pyro.param('a4_mean_{}'.format(index), 0.01 * torch.randn(128 + 1, 2))
    a4_scale = pyro.param('a4_scale_{}'.format(index), 0.1 * torch.ones(128 + 1, 2),
                           constraint=constraints.greater_than(0.01))
        # Sample latent values using the variational parameters that are set-up above.
        # Notice how there is no conditioning on labels in the guide!
    with pyro.plate('data', size=n_X_data):
        h1 = pyro.sample('h1', hnn(X_data, a1_mean,  a1_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
        h2 = pyro.sample('h2', hnn(h1, a2_mean,  a2_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
        h3 = pyro.sample('h3', hnn(h2, a3_mean,  a3_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
        logits = pyro.sample('logits', hnn(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: torch.sigmoid(x),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))


def model(X_data,kl_factor=1.0):
    #images = images.view(-1, 30)
    #print("model")
    features = X_data.shape[1]
    X_data = X_data.view(-1, features)    
    n_X_data = X_data.size(1)
    # Set-up parameters for the distribution of weights for each layer `a<n>`
    a1_mean = torch.zeros(n_X_data, 128)
    a1_scale = torch.ones(n_X_data, 128) - 0.5
    #z_1 = pyro.sample('z', dist.Normal(a1_mean, a1_scale))
    a1_dropout = torch.tensor(0.25)
    #z_1_scale = torch.zeros(30,128) + 0.1
        
    a2_mean = torch.zeros(128 + 1, 128) 
    a2_scale = torch.ones(128 + 1, 128) - 0.5
    #z_2 = pyro.sample('z', dist.Normal(a2_mean, a2_scale))
    a2_dropout = torch.tensor(1.0)
    #z_2_scale = torch.zeros(30,128) + 0.1

    a3_mean = torch.zeros(128 + 1, 128)
    a3_scale = torch.ones(128 + 1, 128) - 0.5
    #z_3 = pyro.sample('z', dist.Normal(a3_mean, a3_scale))
    a3_dropout = torch.tensor(1.0)
    #z_3_scale = torch.zeros(30,128) + 0.1

    a4_mean = torch.zeros(128 + 1, 2)
    a4_scale = torch.ones(128 + 1, 2) - 0.5
    #z_4 = pyro.sample('z', dist.Normal(a4_mean, a4_scale))
    #z_4_scale = torch.zeros(30,128) + 0.1

    # Mark batched calculations to be conditionally independent given parameters using `plate`
    with pyro.plate('data', size=n_X_data):
        # Sample first hidden layer
        h1 = pyro.sample('h1', hnn(X_data, a1_mean,  a1_scale,
                                               non_linearity=nnf.leaky_relu,
                                               KL_factor=kl_factor))
        # Sample second hidden layer
        h2 = pyro.sample('h2', hnn(h1, a2_mean,  a2_scale,
                                               non_linearity=nnf.leaky_relu,
                                               KL_factor=kl_factor))
        # Sample third hidden layer
        h3 = pyro.sample('h3', hnn(h2, a3_mean,  a3_scale,
                                               non_linearity=nnf.leaky_relu,
                                               KL_factor=kl_factor))
        # Sample output logits
        logits = pyro.sample('logits', hnn(h3, a4_mean, a4_scale,
                                                       non_linearity=lambda x: torch.sigmoid(x,dim=-1),
                                                       KL_factor=kl_factor,
                                                       include_hidden_bias=False))
            
        #print(logits,"SVI model")
            

        # One-hot encode labels
        # labels = nnf.one_hot(labels.to(torch.int64)) if labels is not None else None
        # Condition on observed labels, so it calculates the log-likehood loss when training using VI
        # return pyro.sample('lable',dist.OneHotCategorical(logits = logits) , obs = labels)

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


def approximation(data, components, weights):
    assignment = pyro.sample('assignment', dist.Categorical(weights))
    result = components[assignment](data)
    return result


def boosting_bbvi():
    # T=2
    n_iterations = 2
    initial_approximation = partial(guide, index=0)
    #print("initial_approximation",initial_approximation)
    components = [initial_approximation]
    #print("components",components)
    weights = torch.tensor([1.])
    #print("Weights",weights)
    wrapped_approximation = partial(approximation, components=components, weights=weights)
    #print("wrapped_approximation",wrapped_approximation)

    locs = [0]
    scales = [0]
    num_epochs =50
    for i in range(num_epochs):
        #print(i)

        for t in range(1, n_iterations + 1):

            # Create guide that only takes data as argument
            wrapped_guide = partial(guide, index=t)
            losses = []

            adam_params = {"lr": 0.01, "betas": (0.90, 0.999)}
            optimizer = Adam(adam_params)

            # Pass our custom RELBO to SVI as the loss function.
            svi = SVI(model, wrapped_guide, optimizer, loss=relbo) # Wrapped guide is build using partial
            #print(t ,"th iteration ")
            for data, labels in train_loader:
                # Pass the existing approximation to SVI.
                #print(data.shape,"shape of data")
                loss = svi.step(data, approximation=wrapped_approximation) #here we cannot feed the lable data as they are not feeding the data in example mentioned on the website
                losses.append(loss)

            # Update the list of approximation components.
            components.append(wrapped_guide)

            # Set new mixture weight.
            new_weight = 2 / (t + 1)

            # In this specific case, we set the mixture weight of the second component to 0.5.
            if t == 2:
                new_weight = 0.5
            weights = weights * (1-new_weight)
            weights = torch.cat((weights, torch.tensor([new_weight])))

            # Update the approximation
            wrapped_approximation = partial(approximation, components=components, weights=weights)
    #     pred = self.forward(X_data, n_samples=1).mean(0)
    #     # print(pred)
    #     total_loss += loss / len(loader.dataset)
    #     total += labels.size(0)
    #
    #     correct += (pred.argmax(-1) == labels).sum().item()
    #     # print(correct)
    #     # print(total)
    #     param_store = pyro.get_param_store()
    # print(f"[Epoch {i + 1}] loss: {total_loss:.5E} accuracy: {correct / total * 100:.5f}")


# def forward(self, X_data, n_samples=10):
#     res = []
#     for i in range(n_samples):
#         t = poutine.trace(self.guide).get_trace(X_data)
#         res.append(t.nodes['logits']['value'])
#     return torch.stack(res, dim=0)

        # Plot the resulting approximation
        # X = np.arange(-10, 10, 0.1)
        # pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
        # total_approximation = np.zeros(X.shape)
        # for i in range(1, n_iterations + 1):
        #     Y = weights[i].item() * scipy.stats.norm.pdf((X - locs[i]) / scales[i])
        #     pyplot.plot(X, Y)
        #     total_approximation += Y
        # pyplot.plot(X, total_approximation)
        # pyplot.plot(data.data.numpy(), np.zeros(len(data)), 'k*')
        # pyplot.title('Approximation of posterior over z')
        # pyplot.ylabel('probability density')
        # pyplot.show()

if __name__ == '__main__':
    boosting_bbvi()