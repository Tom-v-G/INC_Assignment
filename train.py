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
learning_rate = 0.005
momentum = 0.5
hidden_size = 128
num_iterations = 5000
train_test_ratio = 0.67

train_groups, test_groups = split_data(df, train_test_ratio)
model = RNN(hidden_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
loss_fn = nn.MSELoss()


create_window(train_groups[(1, 2)], 20, 10)

# Stochastic Gradient Descent
rand_store_list = np.random.randint(0, 6, num_iterations)
rand_product_list = np.random.randint(0, 9, num_iterations)
rand_index_list = np.random.randint(window_size + 1, train_groups[(0, 0)].shape[0] - 5, num_iterations) # index for sampling from training set

iterator = 0
while iterator < num_iterations - 1:
    iterator += 1

    #Data preprocessing
    store = rand_store_list[iterator]
    product = rand_product_list[iterator]
    index = rand_index_list[iterator]
    feature, target = create_window(train_groups[(store, product)], index, window_size)
    feature = add_store_and_product(feature, store, product)

    #Training
    model.train()
    target_pred = model(feature)
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
        RMSE = np.sqrt(loss_fn(target_pred, target))
        print(f"Epoch {iterator}: RMSE = {RMSE}")



name = f'Model ws={window_size}, lr={learning_rate}, m={momentum} hs={hidden_size}, ni={num_iterations}, tr={train_test_ratio}.pt'
torch.save(model, 'saved_models/' + name)


