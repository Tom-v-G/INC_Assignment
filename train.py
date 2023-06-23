import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from sklearn.preprocessing import MinMaxScaler
from torchmetrics import MeanAbsolutePercentageError

path = 'train.csv'  # update if train.csv is in a different directory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # train on GPU if possible

# Data preprocessing
df = load_data(path)

# Data scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(df[['number_sold']])
value_column = scaler.transform(df[['number_sold']])
df['number_sold'] = value_column

# Hyperparameters
lookback = 10
learning_rate = 0.0001
hidden_size = 128
num_layers = 1
n_epochs = 500
batch_size = 64
train_test_ratio = 0.67
plot_training_RMSE = False # For checking batch RMSE

# Split data into training and testing group
train_groups, test_groups = split_data(df, train_test_ratio)
# Create lookback windows on the training data
features, targets = create_rolling_windows(train_groups, lookback, 1)

# Construct Dataloader of training data
features, targets = torch.Tensor(features), torch.Tensor(targets)
loader = data.DataLoader(data.TensorDataset(features, targets), shuffle=True, batch_size=batch_size, drop_last=True)

# Initialise RNN
model = RNN(hidden_size, num_layers)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = MeanAbsolutePercentageError()

# Batch Gradient Descent
print('Starting Training')
train_RMSE_list = []
for epoch in range(n_epochs):
    print(f'Epoch: {epoch}')
    # Training
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

    # Validation
    if plot_training_RMSE and (epoch % 1 == 0):
        model.eval()
        with torch.no_grad():
            batch_train_RMSE = []
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                y_batch = torch.squeeze(y_batch)
                batch_train_RMSE.append(np.sqrt(loss_fn(y_pred, y_batch)))
            train_RMSE_list.append(np.mean(batch_train_RMSE))

name = f'Model lb={lookback}, lr={learning_rate}, hs={hidden_size}, epochs={n_epochs}, tr={train_test_ratio}, bs={batch_size}, nl={num_layers}'
torch.save(model, 'saved_models/' + name + '.pt')

if plot_training_RMSE:
    fig, ax = plt.subplots()
    ax.plot(train_RMSE_list, color='green')
    plt.plot()
    plt.savefig(f'{name}.pdf')


