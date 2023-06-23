from model import *

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# train on GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Import model
model_name = 'Model lb=10, lr=0.0001, hs=128, epochs=500, tr=0.67, bs=64, nl=1.pt'
model = torch.load(model_name)  # load your model here

# Set model in evaluation mode
model.eval()

# Create parameter dictionary using dictionary comprehension
model_name_adpt = model_name.split('Model ')[1].split('.pt')[0]
parameters = model_name_adpt.split(", ")
par_dict = {par.split('=')[0]: par.split('=')[1] for par in parameters}

# Read required parameters from model name
lookback = int(par_dict['lb'])
train_test_ratio = float(par_dict['tr'])
batch_size = 1

# Data loading and scaling
df = pd.read_csv("train.csv")
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(df[['number_sold']])
value_column = scaler.transform(df[['number_sold']])
df['number_sold'] = value_column

# Load testing data
train_groups, test_groups = split_data(df, train_test_ratio)

# Score model on each (store, product) pair
scores = {}
for key, test_group in test_groups.items():
    # We want to predict 5 step into the future
    test_features, test_targets = create_rolling_windows({key: test_group}, lookback, 5)
    test_features, test_targets = torch.Tensor(test_features), torch.Tensor(test_targets)
    test_loader = data.DataLoader(data.TensorDataset(test_features, test_targets), shuffle=True, batch_size=batch_size,
                                  drop_last=True)
    mape_score = []
    for feature, target in test_loader:
        predictions = []
        # Predict 5 time steps
        for i in range(5):
            pred_value = model(feature)
            predictions.append(pred_value)
            feat_list = feature.tolist()[0]
            feat_list.pop(0)
            feat_list.append(pred_value.tolist())
            feature = torch.tensor([feat_list])
        predictions = torch.tensor([predictions])
        mape_score.append(mean_absolute_percentage_error(target, predictions))

    scores[key] = mape_score

# save the performance metrics to file
np.savez(f"{model_name}_mape_score.npz", scores=scores)


