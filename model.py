from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error


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


def load_data(path):
    df = pd.read_csv(path, engine='python')
    return df


def split_data(df, train_split):
    print(df.shape)
    gb = df.groupby(["store", "product"]) #create subgroups for each store and product
    groups = {x: gb.get_group(x) for x in gb.groups}
    test_index = int(np.floor(groups[(0, 0)].shape[0] * train_split)) #index on which to split

    train_groups = {x: gb.get_group(x).iloc[0:test_index - 1, :] for x in gb.groups}
    test_groups = {x: gb.get_group(x).iloc[test_index:- 1, :] for x in gb.groups}

    return train_groups, test_groups


def create_window(df, index, window_size):
    '''

    :param df: dataframe
    :param index: index of the timeseries value for which we wish to create a lookback window
    :param window_size: lookback period + 1
    :return: list of values with lenght of the window_size
    '''

    if index < window_size - 1:
        raise ValueError(f'Cannot create lookback window of size {window_size} starting at index {index}')

    return df.loc[:, 'number_sold'].iloc[index - window_size: index]


class RNN(nn.Module):
    def __init__(self, window_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size) #LSTM initialization
        self.linear = nn.Linear(hidden_size, 1) #hidden state to output layer
        self.window_size = window_size

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

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

    def save_model(self, name):
        """
        Saves RNN model to disk
        :param path:
        :return:
        """

        raise NotImplementedError

    '''
        def normalize_data(self, dataset):
            dataset = dataset.reshape(-1, 1)
            dataset = dataset.astype("float32")
            dataset.shape

            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)

            return dataset
    '''
