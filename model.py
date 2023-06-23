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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #train on GPU if possible

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

def create_rolling_windows(df_dict, lookback, target_amt):
    '''
    create windows [i - window_size, i - window_size + 1, ..., i], [i+1, .., i+k] where i+1, .. i+k are targets
    :param df_dict: Pandas dataframe dictionary indexed (store, product)
    :param windows_size: integer
    :return: np.arrays of feature series and target value
    '''

    feature_list = []
    target_list = []
    for group in df_dict:
        df = df_dict[group]
        i = 0
        while i < len(df) - lookback - target_amt -1:
            feature = df.loc[:, 'number_sold'].iloc[i: i+lookback].values.astype('float32')
            target = df.loc[:, 'number_sold'].iloc[i+lookback:i + lookback + target_amt].astype('float32')
            feature_list.append([feature])
            target_list.append([target])
            i += lookback + target_amt + 1
    return np.vstack(feature_list), np.vstack(target_list)

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
    def __init__(self, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers #amt of LSTM modules chained together
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True) #LSTM initialization
        self.linear = nn.Linear(self.hidden_dim, 1) #hidden state to output layer

    def forward(self, x):
        #  Initializing hidden state for first input
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).requires_grad_()
        #  Initializing cell state for first input
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).requires_grad_()

        x = x.unsqueeze(2) # modify data shape

        #Call the LSTM module
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :]) #  convert to output


        return out.squeeze()

    def predict(self, x):
        '''
        Predict 5 timesteps in the future
        :param x:
        :return:
        '''

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
        :param name: Name for the saved model
        :return:
        """



        raise NotImplementedError
