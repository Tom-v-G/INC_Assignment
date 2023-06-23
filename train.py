import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torchmetrics import MeanAbsolutePercentageError

path = 'train.csv'  # update if train.csv is in a different directory


# Hyperparameters
lookback = 15  # change to different values for different lookback windows
learning_rate = 0.005
hidden_size = 128 #amount of nodes in hidden layer
num_layers = 2 #amt of LSTM modules chained together
n_epochs = 350
train_test_ratio = 0.67
batch_size = 64
scaling = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #train on GPU if possible
'''
Steps to take:
1. Load data from file
2. split data in training and testing set per group (dictionary)
For training
3. create lookback windows within training groups and use pd.concat to make 1 dataframe
3a. scale the data
4. use torch dataloader to generate right datashape
5. train neural net using batch training
6. per ... epochs, look at RMSE

For testing:
3. create lookback windows within test groups and use pd.concat to make 1 dataframe
4. use torch dataloader to generate right datashape
5. test neural net using batch training with MAPE error
'''

#Data preprocessing
df = load_data(path)

#Data scaling
if scaling:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(df[['number_sold']])
    value_column = scaler.transform(df[['number_sold']])
    df['number_sold'] = value_column

train_groups, test_groups = split_data(df, train_test_ratio)
features, targets = create_rolling_windows(train_groups, lookback, 1)
test_features, test_targets = create_rolling_windows(test_groups, lookback, 1)

#Construct Dataloader
features, targets = torch.Tensor(features), torch.Tensor(targets)
test_features, test_targets = torch.Tensor(test_features), torch.Tensor(test_targets)

loader = data.DataLoader(data.TensorDataset(features, targets), shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = data.DataLoader(data.TensorDataset(test_features, test_targets), shuffle=True, batch_size=batch_size, drop_last=True)

#Initialise RNN
model = RNN(hidden_size, num_layers)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# loss_fn = nn.MSELoss()
loss_fn = MeanAbsolutePercentageError()

# Batch Gradient Descent
train_RMSE_list = []
#test_RMSE_list = []

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
        # print(f'Features: {X_batch}')
        # print(y_pred)
        # print(y_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Validation
    if epoch % 5 != 0:
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
        '''
        batch_test_RMSE = []
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_pred = model(X_test)
            batch_test_RMSE.append(np.sqrt(loss_fn(y_pred, y_test)))
        print(f'Test RMSE = {np.mean(batch_test_RMSE)}')
        test_RMSE_list.append(np.mean(batch_test_RMSE))
        '''

fig, ax = plt.subplots()
ax.plot(train_RMSE_list, color='green')
#ax.plot(test_RMSE_list, color='blue')
plt.plot()
#plt.show()


name = f'Model lb={lookback}, lr={learning_rate}, hs={hidden_size}, epochs={n_epochs}, tr={train_test_ratio}, \
bs={batch_size}, nl={num_layers}, sc={scaling}'
torch.save(model, 'saved_models/' + name + '.pt')
plt.savefig(name + '.pdf')

