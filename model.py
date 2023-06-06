from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_util

import numpy as np
import pandas as pd

#set random seed
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.rnn_unit = nn.LSTM(input = )
        #self.state = 0
        #self.weights = np.zeros(shape=())

    def load_data(self, path):
        """
        Loads data from a .csv file and stores it in a pandas dataframe
        :param path: string containing path to .csv file
        :return: pandas dataframe containing the dataset
        """
        dataset = pd.read_csv(path, engine='python')

        return dataset

    def normalize_data(self, dataset):
        dataset = dataset.reshape(-1, 1)
        dataset = dataset.astype("float32")
        dataset.shape

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        return dataset

    def train(self, data):
        """
        Trains the recurrent neural network on a dataset
        :param data:
        :return:
        """

        '''
        Initialisation:
        randomize weights.
        '''

        '''
        Forward pass:

        '''

        '''
        Backpropagation:
        unfold RNN in time, use classic backpropagation to calculate loss function for each timestep.
        Take average of weight gradients and update weights accordingly.
        '''
        raise NotImplementedError

    def evaluate(self, data):
        """
        Predicts values using trained RNN
        :param data:
        :return:
        """

        raise NotImplementedError

    def load_model(self, model_data, path):
        """
        Loads model from external dump file
        :param model_data:
        :param path:
        :return:
        """

        raise NotImplementedError

    def save_model(self, path):
        """
        Saves RNN model to disk
        :param path:
        :return:
        """

        raise NotImplementedError