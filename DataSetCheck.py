# created by Pavia Bera
# Check Data sets weight matrix for multivariate

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

def getdata(dataset):

    if dataset == 'cancer':
        file_handler = open("Data Sets/cancer.csv", "r")
        # using read_csv function that reads from a csv file.
        data = pd.read_csv(file_handler, sep=",")
        # closing the file handler
        file_handler.close()

        # creating a dict file
        Diagnosis = {'M': 1, 'B': 0}
        # Traversing through dataframe diagnosis column and writing values where key matches
        data.diagnosis = [Diagnosis[item] for item in data.diagnosis]
        y = data.diagnosis
        # Obtaining Summary Statistics
        # print("This is the data summary statistics of the cancer data table: \n ", data.describe())
        # normalization of data
        data = preprocessing.scale(data)
        # converting to tensor data type
        #data = torch.tensor(data)
        # Shape of the features and labels
        row = data.shape[0]  # 569
        col = data.shape[1]  # 32
        X = data[0:row, 2:col]
        # X = torch.tensor(X)
        # y = torch.tensor(y)
        # print('type:', type(X))
        # y = nnf.one_hot(y.to(torch.int64)) if y is not None else None
        #### Dividing the data into train, test and validation sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)


    elif dataset == 'hepatitis':
        file_handler = open("Data Sets/hepatitis.csv", "r")
        # using read_csv function that reads from a csv file.
        data = pd.read_csv(file_handler, sep=",")
        # closing the file handler
        file_handler.close()

        # creating a dict file
        y = data.out_class
        # Obtaining Summary Statistics
        # print("This is the data summary statistics of the hepatitis data table: \n ", data.describe())
        data = preprocessing.scale(data)
        # converting to tensor data type
        #data = torch.tensor(data)
        # Shape of the features and labels
        row = data.shape[0]  #
        col = data.shape[1]  #
        X = data[0:row, 1:col]

        y = torch.tensor(y) - 1
        #### Dividing the data into train, test and validation sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.5, random_state=1)




    elif dataset == 'diabetes':
        file_handler = open("Data Sets/diabetes.csv", "r")
        # using read_csv function that reads from a csv file.
        data = pd.read_csv(file_handler, sep=",")
        # closing the file handler
        file_handler.close()

        # creating a dict file
        y = data.Outcome
        # Obtaining Summary Statistics
        # print("This is the data summary statistics of the diabetes data table: \n ", data.describe())
        data = preprocessing.scale(data)
        # converting to tensor data type
        data = torch.tensor(data)
        # Shape of the features and labels
        row = data.shape[0]  #
        col = data.shape[1]  #
        X = data[0:row, 0:col - 1]
        #y = torch.tensor(y)
        #### Dividing the data into train, test and validation sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)



    return x_train, x_val, x_test, y_train, y_val, y_test


# Frequentist Neural Network
'''This is a simple neural network with four Hidden layers, one input layer and one output layer layer.
This is a fully connected Neural network
'''


class get_loader_data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def getmodel(inputSize, hiddenNodeSize, outputSize):

    model = nn.Sequential(
        # first hidden layer
        nn.Linear(inputSize, hiddenNodeSize),
        nn.ReLU(),
        # second hidden layer
        nn.Linear(hiddenNodeSize, hiddenNodeSize),
        nn.ReLU(),
        # # third hidden layer
        # nn.Linear(hiddenNodeSize, hiddenNodeSize),
        # nn.ReLU(),
        # output logits
        nn.Linear(hiddenNodeSize, outputSize),
        nn.Sigmoid()
    )

    return model

def train_model(model, optimizer, criterion, train_loader):
    train_loss = 0.0
    model.train()
    number_of_examples = 0
    counter = 0
    #iteration = 0
    # print(len(x_train))
    # print(type(train_loader))
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for x_batch, y_batch in train_loader:

        data = x_batch
        target = y_batch
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # Clear the gradients
        optimizer.zero_grad()
        # Forward Pass
        output = model(data.float())

        target = target.type_as(output)

        number_of_examples += 1
        # Find the Loss
        loss = criterion(output, target.unsqueeze(1))
        # Calculate gradients
        loss.backward()
        # Update Weights
        optimizer.step()
        # Calculate Loss
        train_loss += loss.item() * data.size(0)


        if output[0] > 0.7:
            out_for_acc = 1.0
        else:
            out_for_acc = 0.0

        if (out_for_acc == target[0]):
            counter += 1
        #accs.append(accuracy_score(output.detach(), target))
    acc = counter / number_of_examples * 100
    #iteration += 1

    return train_loss, acc

def validate_model(model, criterion, validation_loader):
    valid_loss = 0.0
    model.eval()
    # number_of_examples = 0
    # counter = 0
    accs = []
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for x_batch ,y_batch in validation_loader:
        data = x_batch
        target = y_batch
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # Forward Pass
        output = model(data.float())
        output = output.round()
        target = target.type_as(output)
        # Find the Loss
        loss = criterion(output, target.unsqueeze(1))
        valid_loss += loss.item() * data.size(0)
        accs.append(accuracy_score(output.detach(), target))
        # if output[0] > 0.8:
        #     out_for_acc = 1.0
        # else:
        #     out_for_acc = 0.0
        #
        # if (out_for_acc == target[0]):
        #     counter += 1

        # acc = counter / number_of_examples * 100
    return valid_loss, np.mean(accs)

def test_model(model, test_loader):
    model.eval()
    number_of_examples = 0
    counter = 0
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for x_batch, y_batch in test_loader:
        data = x_batch
        target = y_batch
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output = model(data.float())
        target = target.type_as(output)
        # output = output.round()
        number_of_examples += 1

        if output[0] > 0.8:
            out_for_acc = 1.0
        else:
            out_for_acc = 0.0

        if (out_for_acc == target[0]):
            counter += 1

        #accs.append(accuracy_score(output.detach(), target))
    acc = counter / number_of_examples * 100

    return acc

def frequentist(x_train, x_val, x_test, y_train, y_val, y_test):
    # Hyper Parameter settings
    n_epochs = 50
    lr = 0.01
    batch_size = 10
    input_size = x_train.shape[1]
    print('input_size', input_size)
    hidden_node_size = 5
    output_size = 1
    min_valid_loss = np.inf
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = getmodel(input_size, hidden_node_size, output_size)

    if torch.cuda.is_available():
        model = model.cuda()

    # save data
    ckpt_dir = f'checkpoints/frequentist'
    ckpt_name = f'checkpoints/frequentist/model_{model}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.BCELoss()   # Binary cross entropy loss
    optimizer = optim.SGD(model.parameters(), lr=lr)


    train_loader = get_loader_data(torch.FloatTensor(x_train.float()),
                                   torch.FloatTensor(y_train.float()))
    train_loader = DataLoader(dataset=train_loader, shuffle=True)
    val_loader = get_loader_data(torch.FloatTensor(x_val.float()),
                                   torch.FloatTensor(y_val.float()))
    val_loader = DataLoader(dataset=val_loader, shuffle=True)

    test_loader = get_loader_data(torch.FloatTensor(x_test.float()),
                                  torch.FloatTensor(y_test.float()))
    test_loader = DataLoader(dataset=test_loader,  shuffle=True)

    weight_zero = []
    weight_one = []
    weight_two = []
    weight_three = []
    weight_four = []
    train_accu = []
    train_loss_plot = []
    validation_loss =[]
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    for epoch in range(1, n_epochs + 1):
        '''Calling method to visualize the weights'''
        # if (epoch == 50):

        train_loss, acc = train_model(model, optimizer, criterion, train_loader)
        valid_loss, valid_acc = validate_model(model, criterion, val_loader)
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(val_loader)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining acc: {:.4f} '.format(epoch, train_loss, acc),'\t Validation Loss: {:.4f} \t Validation Accuracy: {:.4f}'.format(valid_loss, valid_acc))
        test_acc = test_model(model, test_loader)
        print('Test Accuracy', test_acc)

        if int(test_acc)  in range(90,93):
            SavedWeights = 'SavedWeights'
            torch.save(model.state_dict(), os.path.join(SavedWeights, 'saved_model_90_92-{}.pth'.format(count)))
            count += 1
            print('In 90-93')
        elif int(test_acc) in range(93, 96):
            SavedWeights = 'SavedWeights'
            torch.save(model.state_dict(), os.path.join(SavedWeights, 'saved_model_93_95-{}.pth'.format(count1)))
            count1+= 1
            print('In 93-96')
        elif int(test_acc) in range(96, 99):
            SavedWeights = 'SavedWeights'
            torch.save(model.state_dict(), os.path.join(SavedWeights, 'saved_model_96_98-{}.pth'.format(count2)))
            count2+= 1
            print('In 96-99')
        elif int(test_acc) in range(99, 101):
            SavedWeights = 'SavedWeights'
            torch.save(model.state_dict(), os.path.join(SavedWeights, 'saved_model_99_100-{}.pth'.format(count3)))
            count3 += 1
            print('In 99-101')

        acc = np.array(acc)
        train_loss = np.array(train_loss)
        valid_loss = np.array(valid_loss)

        train_accu.append(acc)
        train_loss_plot.append(train_loss)
        validation_loss.append(valid_loss)
        # if min_valid_loss > valid_loss:
        #     #print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        #     min_valid_loss = valid_loss

            # Saving State Dict
        #torch.save(model.state_dict(), 'saved_model.pth')
        #SavedWeights = 'SavedWeights'
        #torch.save(model.state_dict(), os.path.join(SavedWeights, 'saved_model_epoch-{}.pth'.format(epoch)))

    #     #visualize
    #     for name, param in model.named_parameters():
    #         if (name == '4.weight'):
    #             weights = param.detach().numpy()
    #             weights = np.array(weights)
    #             #print(weights)
    #             weight_zero.append(weights[0, 0])
    #
    # print('weight_zero: ', np.array(weight_zero).shape)
    # # plot weights
    # plt.plot(weight_zero, '*')
    # #plt.title('For epoch : {}'.format(epoch))
    # plt.title('Weight 3,1')
    # plt.show()
    # test_acc = test_model(model, test_loader)
    # print('Testing acc: {:.4f} '.format(test_acc))

    #
    # ### PLot Train loss and Train accuracy
    # plt.plot(train_accu)
    # plt.plot(train_loss_plot, '-')
    # plt.plot(validation_loss)
    # #plt.title('For epoch : {}'.format(epoch))
    # plt.title('Accuracy and Loss')
    # plt.legend(["train_accu", "train_loss", "Validation loss"])
    # plt.show()

    visualize(model)




def visualize(model):

    # model_new = model
    # model_new.load_state_dict(torch.load('saved_model.pth'))

    weight_zero = []
    weight_one = []
    weight_two = []
    weight_three = []
    weight_four = []
    # Printing all the parameters of the model
    #print("For epoch : ", epoch)
    for name, param in model.named_parameters():
        print('name: ', name)
        print(type(param))
        print('param.shape: ', param.shape)
        print('param.requires_grad: ', param.requires_grad)
        print('=====')


        # if (name == '4.weight'):
        #     #weights.append((param.detach().numpy()))
        #     #print(param.detach().numpy())
        #     weights = param.detach().numpy()
        #     weights = np.array(weights)
        #     weight_zero.append(weights[0,0])




    #print('weight shape: ', weights.shape)
    # node_number = weights.shape[1]
    # weight_number =  weights.shape[2]
    #
    #print('weight: ', weight_zero)
    # plot weights
    # plt.plot(weight_zero, '*')
    # plt.title('For epoch : {}'.format(epoch))
    # plt.show()










def datasettype(dataset):
    # Get data set
    x_train, x_val, x_test, y_train, y_val, y_test = getdata(dataset)


    x_train = torch.Tensor(x_train)
    x_val = torch.Tensor(x_val)
    x_test = torch.Tensor(x_test)
    y_train = torch.Tensor(y_train.values)
    y_val = torch.Tensor(y_val.values)
    y_test = torch.Tensor(y_test.values)

    frequentist(x_train, x_val, x_test, y_train, y_val, y_test)


# Driver Code
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Frequentist Training")
    parser.add_argument('--dataset', default='cancer', type=str, help='dataset = [cancer/hepatitis/diabetes]')
    args = parser.parse_args()
    datasettype(args.dataset)
