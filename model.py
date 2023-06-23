import torch
import torch.nn as nn

import numpy as np
import pandas as pd

# Set random seed
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

# Train on GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(path):
    # Reads data from file and drops the data column
    df = pd.read_csv(path, engine='python')
    df = df.drop(["Date"], axis=1)
    return df


def split_data(df, train_split):
    # Split data into a training group and a test group
    gb = df.groupby(["store", "product"])  # Create subgroups for each store and product
    groups = {x: gb.get_group(x) for x in gb.groups}
    test_index = int(np.floor(groups[(0, 0)].shape[0] * train_split))  # Index on which to split

    train_groups = {x: gb.get_group(x).iloc[0:test_index - 1, :] for x in gb.groups}
    test_groups = {x: gb.get_group(x).iloc[test_index:- 1, :] for x in gb.groups}

    return train_groups, test_groups


def create_rolling_windows(df_dict, lookback, target_amt):
    # Window all data in a dataframe dictionary, windows have length lookback + target_amt
    # Where lookback is how many entries we observe and target_amt is how many instances we predict
    # Returns as numpy array

    feature_list = []
    target_list = []
    for group in df_dict:
        df = df_dict[group]
        i = 0
        while i < len(df) - lookback - target_amt - 1:
            feature = df.loc[:, 'number_sold'].iloc[i: i+lookback].values.astype('float32')
            target = df.loc[:, 'number_sold'].iloc[i+lookback:i + lookback + target_amt].astype('float32')
            feature_list.append([feature])
            target_list.append([target])
            i += lookback + target_amt + 1
    return np.vstack(feature_list), np.vstack(target_list)


class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_size  # Amt of hidden nodes
        self.num_layers = num_layers  # Amt of LSTM modules chained together
        # LSTM initialization
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, 1)  # Hidden state to output layer

    def forward(self, x):
        # Initializing hidden state for first input
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).requires_grad_()
        # Initializing cell state for first input
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).requires_grad_()
        # Modify data shape
        x = x.unsqueeze(2)
        # Call the LSTM module
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Convert to output
        out = self.linear(out[:, -1, :])

        return out.squeeze()


