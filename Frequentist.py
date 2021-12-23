# Frequentist Neural Network
'''This is a simple neural network with four Hidden layers, one input layer and one output layer layer.
This is a fully connected Neural network
'''
from collections import OrderedDict
from sklearn import model_selection
import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score


# import Network

class get_loader_data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__ (self):
        return len(self.X_data)

class getmodel_01(nn.Module):
    # define model architecture 1
    def __init__(self ,inputSize,hiddenNodeSize,outputSize):
        super(getmodel_01,self).__init__()
        self.model = nn.Sequential(
            # first hidden layer
            nn.Linear(inputSize, hiddenNodeSize),
            nn.ReLU(),
            # second hidden layer
            nn.Linear(hiddenNodeSize, hiddenNodeSize),
            nn.ReLU(),
            # third hidden layer
            nn.Linear(hiddenNodeSize, hiddenNodeSize),
            nn.ReLU(),
            #output logits
            nn.Linear(hiddenNodeSize, outputSize),
            nn.Sigmoid()
            )
        #summary = self.model.models()
        #print(self.summary)


class getmodel_02(nn.Module):
    # define model architecture 2
    def __init__(self ,inputSize,hiddenNodeSize,outputSize):
        super(getmodel_02,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputSize, hiddenNodeSize),
            nn.LeakyReLU(), #todo leaky_relu
            nn.Linear(hiddenNodeSize, hiddenNodeSize),
            nn.LeakyReLU(),
            nn.Linear(hiddenNodeSize, hiddenNodeSize),
            nn.LeakyReLU(),
            nn.Linear(hiddenNodeSize, outputSize),
            nn.Sigmoid()
            )
        #summary = self.model.models()
        #print(self.summary)


class Frequentist():

    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test):
        # converting to tensor data type
        # print(x_train.shape)
        # print(type(x_train))
        # x_train = tf.convert_to_tensor(x_train)
        # x_val = tf.convert_to_tensor(x_val)
        # x_test = tf.convert_to_tensor(x_test)
        # y_train = tf.convert_to_tensor(y_train)
        # y_val = tf.convert_to_tensor(y_val)
        # y_test = tf.convert_to_tensor(y_test)
        print("This is a plain vanilla NN")
        # Hyper Parameter settings
        n_epochs = 50
        lr = 0.002
        
        batch_size = 1
        self.inputSize = x_train.shape[1]
        self.hiddenNodeSize = 128
        self.outputSize = 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net_type = 'Model_01'
        # print(train_loader)
        net_1 = getmodel_01(self.inputSize, self.hiddenNodeSize, self.outputSize)
        net_2 = getmodel_02(self.inputSize, self.hiddenNodeSize, self.outputSize)

        # save data
        ckpt_dir = f'checkpoints/frequentist'
        ckpt_name = f'checkpoints/frequentist/model_{net_type}.pt'

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(net_1.parameters(), lr=lr)

        train_loader = get_loader_data(torch.FloatTensor(x_train.float()), 
                       torch.FloatTensor(y_train.float()))
        train_loader = DataLoader(dataset=train_loader, batch_size=batch_size, shuffle=True)

        test_loader = get_loader_data(torch.FloatTensor(x_test.float()), 
                       torch.FloatTensor(y_test.float()))
        test_loader = DataLoader(dataset=test_loader, batch_size=10, shuffle=True)

        for epoch in range(1, n_epochs + 1):

            train_loss ,acc = self.train_model(net_1, optimizer, criterion, train_loader )
           
            train_loss = train_loss / len(x_train)
            #valid_loss = valid_loss / len(x_train)

            print('Epoch: {} \tTraining Loss: {:.4f} \tTraining acc: {:.4f} '.format(epoch, train_loss ,acc))
                    # \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'valid_loss, valid_acc))
        
        test_acc = self.test_model(net_1,test_loader)
        print('Testing acc: {:.4f} '.format(test_acc))

    def train_model(self, net, optimizer, loss_fn, train_loader):
        train_loss = 0.0
        net.train()
        number_of_examples = 0
        counter = 0
        #print(len(x_train))
        #print(type(train_loader))
        for x_batch ,y_batch in train_loader:
            data = x_batch
            target = y_batch
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = net.model(data.float())
            
            target = target.type_as(output)
            #output = output.round()
            
            number_of_examples += 1
            loss = loss_fn(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            if output[0] > 0.8 :
                out_for_acc = 1.0
            else:
                out_for_acc = 0.0
           
            if(out_for_acc == target[0]):
                counter += 1
            #accs.append(accuracy_score(output.detach(), target))
        acc = counter/number_of_examples *100
        return train_loss , acc

    def test_model(self, net,test_loader):
        net.eval()
        number_of_examples = 0
        counter = 0
        
        for x_batch ,y_batch in test_loader:
            data = x_batch
            target = y_batch
            data, target = data.to(self.device), target.to(self.device)
            
            output = net.model(data.float())
            target = target.type_as(output)
            #output = output.round()
            number_of_examples += 1
            
            if output[0] > 0.8 :
                out_for_acc = 1.0
            else:
                out_for_acc = 0.0

            
            if(out_for_acc == target[0]):
                counter += 1

            #accs.append(accuracy_score(output.detach(), target))
        acc = counter/number_of_examples *100

        return acc

    # def validate_model(self, net, criterion, validation_loader):
    #     valid_loss = 0.0
    #     net.eval()
    #     accs = []
    #     for x_batch ,y_batch in validation_loader:
    #         data = x_batch
    #         target = y_batch
    #         data, target = data.to(self.device), target.to(self.device)
    #         output = net.model(data.float())
    #         output = output.round()
    #         target = target.type_as(output)
    #         loss = criterion(output, target.unsqueeze(1))
    #         valid_loss += loss.item() * data.size(0)
    #         accs.append(accuracy_score(output.detach(), target))
    #     return valid_loss, np.mean(accs)





        print("Finished training!")



######################################################################################################

