# Importing all the necessary libraries and my other classes.
import argparse
import parser
import numpy as np
import argparse
from sklearn import preprocessing
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import DataSetCheck
from itertools import chain

def weightdistribution():
    # input_size = DataSetCheck.input_size
    # print(input_size)
    input_size = 30
    hidden_node_size = 5
    output_size = 1
    n_epochs = 50
    weights = []
    model_new = DataSetCheck.getmodel(input_size, hidden_node_size, output_size)

    for epoch in range(1, n_epochs + 1):
        if epoch == 10:
            model_new.load_state_dict(torch.load(os.path.join('SavedWeights/saved_model_epoch-{}.pth'.format(epoch))))

    for name, param in model_new.named_parameters():
        print(name, ':', param.requires_grad)

    # visualize
    for name, param in model_new.named_parameters():
        if (name == '0.weight'):
            w = param.detach().numpy()
            # w = np.array(w)
            print('Input Layer', w.shape)
            for i in range(w.shape[0]):
                weights.append(w[i, :])
            print('row', len(weights))
            print('Col', len(weights[0]))
        elif (name == '2.weight'):
            w = param.detach().numpy()
            # w = np.array(w)
            print('Hidden Layer', w.shape)
            for i in range(w.shape[0]):
                weights.append(w[i, :])
            print('row', len(weights))
            print('Col', len(weights[0]))
        elif (name == '4.weight'):
            w = param.detach().numpy()
            # w = np.array(w)
            print('Last layer', w.shape)
            for i in range(w.shape[0]):
                weights.append(w[i, :])
            print('row', len(weights))
            print('Col', len(weights[0]))

    # converting to list
    weight = [list(x) for x in weights]
    # Flat list
    flat_list = []
    for sublist in weight:
        for item in sublist:
            flat_list.append(item)

    # print(flat_list)
    # plot weights
    plt.hist(flat_list, 10)
    plt.title('Weight Distribution over 10 Epoch')
    plt.xlabel('Bins')
    plt.ylabel('Weight Value')
    plt.show()

#########################################################################################################
# input_size = DataSetCheck.input_size
# print(input_size)
input_size = 30
hidden_node_size = 5
output_size = 1
n_epochs = 50
weights = []
model_new = DataSetCheck.getmodel(input_size, hidden_node_size, output_size)

# for test_acc in range(90,101):
    # if int(test_acc)  in range(90,93):
    #     for count in range()
    #     model_new.load_state_dict(torch.load(os.path.join('SavedWeights/saved_model_90_92-{}.pth'.format(count))))
    #
    # elif int(test_acc) in range(93, 96):

    # if int(test_acc) in range(96,99):
model_new.load_state_dict(torch.load(os.path.join('SavedWeights/saved_model_96_98-{}.pth'.format(0))))
for name, param in model_new.named_parameters():
    if (name == '0.weight'):
        w = param.detach().numpy()
        n0 = w.shape

    elif (name == '2.weight'):
        w = param.detach().numpy()
        n2 = w.shape

    elif (name == '4.weight'):
        w = param.detach().numpy()
        n4 = w.shape


## Initializing all the weights
n0range = n0[0]*n0[1]
weight_zero = [[] for x in range(n0range)]
n2range = n2[0]*n2[1]
weight_two = [[] for x in range(n2range)]
n4range = n4[0]*n4[1]
weight_four = [[] for x in range(n4range)]
# print(weight_zero)
for count in range(0, 4):
    model_new.load_state_dict(torch.load(os.path.join('SavedWeights/saved_model_93_95-{}.pth'.format(count))))
    # visualize
    for name, param in model_new.named_parameters():
        if (name == '0.weight'):
            w = param.detach().numpy()
            # w = np.array(w)
            #print('Input Layer', w.shape)
            x = 0
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    weight_zero[x].append(w[i][j])
                    x += 1

        elif (name == '2.weight'):
            w = param.detach().numpy()
            # w = np.array(w)
            #print('Hidden Layer', w.shape)
            x = 0
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    weight_two[x].append(w[i][j])
                    x += 1
                    # print(x)
        elif (name == '4.weight'):
            w = param.detach().numpy()
            # w = np.array(w)
            #print('Last layer', w.shape)
            x = 0
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    weight_four[x].append(w[i][j])
                    x += 1

# # plot weights
#fig = plt.figure(figsize=(5, 12))
for i in range(0, len(weight_zero)):
    plt.hist(weight_zero[i], 5)
    plt.title('Weight Distribution in range 93 to 95 for layer 0 weight {}'.format(i))
    plt.xlabel('Bins')
    plt.ylabel('Weight Value')
    plt.savefig('Layer 0 weight {}'.format(i))
    plt.clf()
for i in range(0, len(weight_two)):
    plt.hist(weight_two[i], 5)
    plt.title('Weight Distribution in range 93 to 95 for layer 1 weight {}'.format(i))
    plt.xlabel('Bins')
    plt.ylabel('Weight Value')
    plt.savefig('Layer 1 weight {}'.format(i))
    plt.clf()
for i in range(0, len(weight_four)):
    plt.hist(weight_four[i], 5)
    plt.title('Weight Distribution in range 93 to 95 for layer 2 weight {}'.format(i))
    plt.xlabel('Bins')
    plt.ylabel('Weight Value')
    plt.savefig('Layer 2 weight {}'.format(i))
    plt.clf()