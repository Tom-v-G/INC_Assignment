from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_util
from torch.autograd import Variable

import numpy as np
import pandas as pd

#set random seed
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)


def load_data(path):
    df = pd.read_csv(path, engine='python')
    df = df.drop(["Date"], axis=1) #remove date from dataframe

    return df


def split_data(df, train_split):
    gb = df.groupby(["store", "product"]) #create subgroups for each store and product
    groups = {x: gb.get_group(x) for x in gb.groups}
    test_index = int(np.floor(groups[(0, 0)].shape[0] * train_split)) #index on which to split

    train_groups = {x: gb.get_group(x).iloc[0:test_index - 1, :] for x in gb.groups}
    test_groups = {x: gb.get_group(x).iloc[test_index:- 1, :] for x in gb.groups}

    return train_groups, test_groups


def scale_and_window(df, index, window_size):
    '''
    Create a lookback window (feature) and a prediction window (target) of a timeseries in a dataframe

    :param df: dataframe
    :param index: index of the timeseries value for which we wish to create a lookback window
    :param window_size: lookback period + 1
    :return: feature list of timeseries values with length of the window_size and target list of values with length 5
    as tensors.
    '''
    #scaler = MinMaxScaler(feature_range=(0, 1))

    if index < window_size - 1:
        raise ValueError(f'Cannot create lookback window of size {window_size} starting at index {index}.')
    if index + 6 > df.shape[0]:
        raise ValueError(f'Index {index + 6} is out of bounds.')

    feature = df.loc[:, 'number_sold'].iloc[index - window_size+1: index+1].values.astype('float32')
    target = df.loc[:, 'number_sold'].iloc[index + 1: index + 6].values.astype('float32')

    return torch.tensor(feature), torch.tensor(target)


def add_store_and_product(feature, store, product):
    '''
    Add the store number and product number to the feature tensor
    :param feature: tensor containing timeseries feature of an instance
    :param store: store number
    :param product: product number
    :return: multidimensional tensor of the form [feature, store, product]
    '''
    store_tensor = torch.tensor(np.full(len(feature), store))
    product_tensor = torch.tensor(np.full(len(feature), product))
    updated_feature = torch.stack((feature, store_tensor, product_tensor), dim=1)
    return updated_feature

class RNN(nn.Module):
    def __init__(self, hidden_size, input_amt):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = 1 #amt of LSTM modules chained together
        self.lstm = nn.LSTM(input_size=input_amt, hidden_size=self.hidden_dim) #LSTM initialization
        self.linear = nn.Linear(self.hidden_dim, 5) #hidden state to output layer

    def forward(self, x):
        #  Initializing hidden state for first input
        h0 = torch.zeros(1, self.hidden_dim).requires_grad_()
        #  Initializing cell state for first input
        c0 = torch.zeros(1, self.hidden_dim).requires_grad_()

        #Call the LSTM module
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[-1, :]) #  convert to output
        return out

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
        :param name: Name for the saved model
        :return:
        """



        raise NotImplementedError
