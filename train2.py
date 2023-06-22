import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

path = 'train.csv'  # update if train.csv is in a different directory
df = load_data(path)

# Hyperparameters
window_size = 10  # change to different values for different lookback windows
learning_rate = 0.05
momentum = 0.3
hidden_size = 256
num_iterations = 20000
train_test_ratio = 0.8

train_groups, test_groups = split_data(df, train_test_ratio)
model = RNN(hidden_size, 1)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
loss_fn = nn.MSELoss()

# Stochastic Gradient Descent
rand_store_list = np.random.randint(0, 6, num_iterations)
rand_product_list = np.random.randint(0, 9, num_iterations)
rand_index_list = np.random.randint(window_size + 1, train_groups[(0, 0)].shape[0] - 5, num_iterations) # index for sampling from training set

iterator = 0

RMSE_list = []
while iterator < num_iterations - 1:
    iterator += 1

    #Data preprocessing
    store = rand_store_list[iterator]
    product = rand_product_list[iterator]
    index = rand_index_list[iterator]

    feature, target_scaled = scale_and_window(train_groups[(store, product)], index, window_size)
    #feature = add_store_and_product(feature_scaled, store, product)

    #Training
    model.train()
    target_pred = model(feature)
    #target_pred_scaled = model(feature)

    #Invert scaling
    #target_pred = scaler.inverse_transform(target_pred_scaled)
    target = target_scaled
    #target = scaler.inverse_transform(target_scaled)

    loss = loss_fn(target_pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Validation
    if iterator % 100 != 0:
        continue

    model.eval()
    with torch.no_grad():
        target_pred = model(feature)
        # target_pred_scaled = model(feature)
        #target_pred = scaler.inverse_transform(target_pred_scaled)
        target = target_scaled
        #target = scaler.inverse_transform(target_scaled)
        print(f"Epoch: {iterator}")
        print('target: ', target)
        print('prediction: ', target_pred)
        RMSE = np.sqrt(loss_fn(target_pred, target))
        print(f'RMSE = {RMSE}')
        RMSE_list.append(RMSE)

fig, ax = plt.subplots()
ax.plot(RMSE_list)
plt.plot()
plt.show()


name = f'Model ws={window_size}, lr={learning_rate}, m={momentum} hs={hidden_size}, ni={num_iterations}, tr={train_test_ratio}.pt'
torch.save(model, 'saved_models/' + name)


