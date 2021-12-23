# created by Pavia Bera
# Neural Network weight training with Variational Inference and Boosting
# NCRG labs
# University of South Florida
'''This is the main Class from where all other classes will be called'''
# ---------------------------------------------------------------------------------------------------------------------------#

# Importing all the necessary libraries and my other classes.
import argparse

from sklearn.model_selection import train_test_split
from Bayesian import Bayesian
import torch

import numpy as np
import matplotlib
from sklearn import preprocessing
import pandas as pd

from Frequentist import Frequentist
from boosting import Boosting
import torch.nn.functional as nnf

def getdata(dataset):

    # get the data class domain
    # Importing Data Sets
    # creating file handler for our cancer.csv file in read mode
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
        print("This is the data summary statistics of the data table: \n ", data.describe())
        # normalization of data
        data = preprocessing.scale(data)
        # converting to tensor data type
        data = torch.tensor(data)
        # Shape of the features and labels
        row = data.shape[0]  # 569
        col = data.shape[1]  # 32
        X = data[0:row, 2:col]
        y = torch.tensor(y)
        y = nnf.one_hot(y.to(torch.int64)) if y is not None else None
        print('y: ', type(y))
        #### Dividing the data into train, test and validation sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)


    elif dataset == 'hepatitis':
        file_handler = open("/home/pranav/BVI_NN/Data Sets/hepatitis.csv", "r")
        # using read_csv function that reads from a csv file.
        data = pd.read_csv(file_handler, sep=",")
        # closing the file handler
        file_handler.close()

        # creating a dict file
        y = data.out_class
        # Obtaining Summary Statistics
        print("This is the data summary statistics of the data table: \n ", data.describe())
        data = preprocessing.scale(data)
        # converting to tensor data type
        data = torch.tensor(data)
        # Shape of the features and labels
        row = data.shape[0]  #
        col = data.shape[1]  #
        X = data[0:row, 1:col]

        y = torch.tensor(y) - 1
        #### Dividing the data into train, test and validation sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.5, random_state=1)




    elif dataset == 'diabetes':
        file_handler = open("/home/pranav/BVI_NN/Data Sets/diabetes.csv", "r")
        # using read_csv function that reads from a csv file.
        data = pd.read_csv(file_handler, sep=",")
        # closing the file handler
        file_handler.close()

        # creating a dict file
        y = data.Outcome
        # Obtaining Summary Statistics
        print("This is the data summary statistics of the data table: \n ", data.describe())
        data = preprocessing.scale(data)
        # converting to tensor data type
        data = torch.tensor(data)
        # Shape of the features and labels
        row = data.shape[0]  #
        col = data.shape[1]  #
        X = data[0:row, 0:col-1]
        # y = torch.tensor(y)
        #### Dividing the data into train, test and validation sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)



    #### Dividing the data into train, test and validation sets


    return x_train, x_val, x_test, y_train, y_val, y_test


def procedure(dataset, training_method):

    # Get data set
    x_train, x_val, x_test, y_train, y_val, y_test = getdata(dataset)
    #print(x_train.shape)

    # Call proper training Method
    if training_method == 'frequentist':
        print("Calling Frequentist Class")
        # Creating an instance of Frequentist class

        return Frequentist(x_train, x_val, x_test, y_train, y_val, y_test)

    elif training_method == 'VI':
        print("Calling Variational Inference Class")

        return Bayesian(x_train, x_val, x_test, y_train, y_val, y_test)

    elif training_method == 'BVI':
        print("Calling Boosting_Of_VI Class")

        return Boosting(x_train, x_val, x_test, y_train, y_val, y_test)





###############################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Frequentist/Bayesian/Boosting Model Training")
    parser.add_argument('--dataset', default='cancer', type=str, help='dataset = [cancer/hepatitis/diabetes]')
    parser.add_argument('--method', default='frequentist', type=str, help='method = [frequentist/VI/BVI]')
    args = parser.parse_args()
    procedure(args.dataset, args.method)

# calling network class
#my_MainClass_instance = MainClass()
#my_MainClass_instance.getdata()
#my_MainClass_instance.procedure()




