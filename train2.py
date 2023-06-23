import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torchmetrics import MeanAbsolutePercentageError

path = 'train.csv'  # update if train.csv is in a different directory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #train on GPU if possible

#Data preprocessing
df = load_data(path)

#Data scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(df[['number_sold']])
value_column = scaler.transform(df[['number_sold']])
df['number_sold'] = value_column

# Hyperparameters
lookback = 15  # change to different values for different lookback windows

hidden_size = 128 #amount of nodes in hidden layer
num_layers = 2 #amt of LSTM modules chained together
#
# lookback_list = [10, 10, 10, 15, 15, 15, 15, 20, 20, 20]
# learning_rate_list = [0.005, 0.001, 0.001, 0.005, 0.005, 0.001, 0.001, 0.005, 0.001, 0.001]
# hidden_size_list = [64, 128, 256, 64, 128, 256, 512, 64, 128, 256]
# num_layers_list = [1, 1, 2, 1, 1, 2, 2, 1, 1, 2]
# n_epochs = 20
# batch_size = 64
# train_test_ratio = 0.67
# learning_rate = 0.0001

lookback_list = [10]
learning_rate_list = [0.0001]
hidden_size_list = [128]
num_layers_list = [1]
n_epochs = 500
batch_size = 64
train_test_ratio = 0.67

for lookback, learning_rate, hidden_size, num_layers in zip(lookback_list, learning_rate_list, hidden_size_list, num_layers_list):
    train_groups, test_groups = split_data(df, train_test_ratio)
    features, targets = create_rolling_windows(train_groups, lookback, 1)
    test_features, test_targets = create_rolling_windows(test_groups, lookback, 1)

    # Construct Dataloader
    features, targets = torch.Tensor(features), torch.Tensor(targets)
    test_features, test_targets = torch.Tensor(test_features), torch.Tensor(test_targets)

    loader = data.DataLoader(data.TensorDataset(features, targets), shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = data.DataLoader(data.TensorDataset(test_features, test_targets), shuffle=True, batch_size=batch_size,
                                  drop_last=True)

    #Initialise RNN
    model = RNN(hidden_size, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # loss_fn = nn.MSELoss()
    loss_fn = MeanAbsolutePercentageError()

    # Batch Gradient Descent

    print('Starting Training')
    train_RMSE_list = []
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch}')
        #Training
        model.train()
        batch_RMSE = []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            y_batch = torch.squeeze(y_batch)

            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #Validation
        if epoch % 10 != 0:
            continue

        model.eval()
        with torch.no_grad():
            batch_train_RMSE = []
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                y_batch = torch.squeeze(y_batch)
                batch_train_RMSE.append(np.sqrt(loss_fn(y_pred, y_batch)))
            #print(f'Train RMSE = {np.mean(batch_train_RMSE)}')
            train_RMSE_list.append(np.mean(batch_train_RMSE))

    fig, ax = plt.subplots()
    ax.plot(train_RMSE_list, color='green')
    plt.plot()


    name = f'Model lb={lookback}, lr={learning_rate}, hs={hidden_size}, epochs={n_epochs}, tr={train_test_ratio}, bs={batch_size}, nl={num_layers}'
    torch.save(model, 'saved_models/' + name + '.pt')
    plt.savefig('plots/' + name + '.pdf')

