#
# Data visualisation
#a

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

from model import *

def visualise_data():
    test = RNN()
    df = load_data('train.csv')

    gb = df.groupby(["store", "product"])
    groups = {x: gb.get_group(x) for x in gb.groups}

    fig, axs = plt.subplots(7, 10, figsize=(40, 35))
    for index, store_axs in enumerate(axs):
        for index2, product_ax in enumerate(store_axs):
            X = groups[(index, index2)]
            X = X.drop(["store", "product"], axis=1).values
            X = [item for sublist in X for item in sublist]
            product_ax.plot(X, linewidth=0.5)
            product_ax.set_yticks([min(X), max(X)])
            if index == 6:
                product_ax.set_xticks([0, len(X)])
            else:
                product_ax.set_xticks([])
            product_ax.tick_params(labelsize=6)

    for index, ax in enumerate(axs[0]):
        ax.set_title(f"Product {index}")

    for index, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(f"Store {index}", rotation=0, size='large', labelpad=20)

    plt.subplots_adjust(wspace=0.65, hspace=0.3)
    #fig.tight_layout()
    plt.show()

def visualize_training_groups():
    df = load_data('train.csv')
    train_groups, test_groups = split_data(df, 0.8)

    fig, axs = plt.subplots(7, 10, figsize=(40, 35))
    for index, store_axs in enumerate(axs):
        for index2, product_ax in enumerate(store_axs):
            X = train_groups[(index, index2)]
            X = X.drop(["store", "product"], axis=1).values
            X = [item for sublist in X for item in sublist]
            product_ax.plot(X, linewidth=0.5)
            product_ax.set_yticks([min(X), max(X)])
            if index == 6:
                product_ax.set_xticks([0, len(X)])
            else:
                product_ax.set_xticks([])
            product_ax.tick_params(labelsize=6)

    for index, ax in enumerate(axs[0]):
        ax.set_title(f"Product {index}")

    for index, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(f"Store {index}", rotation=0, size='large', labelpad=20)

    plt.subplots_adjust(wspace=0.65, hspace=0.3)
    # fig.tight_layout()
    plt.show()

def visualise_prediction():
    model_name = 'Model lb=10, lr=0.0001, hs=128, epochs=500, tr=0.67, bs=64, nl=1.pt'  #
    model = torch.load('saved_models/' + model_name)  # load your model here
    model.eval()

    model_name_adpt = model_name.split('Model ')[1].split('.pt')[0]
    parameters = model_name_adpt.split(", ")
    par_dict = {par.split('=')[0]: float(par.split('=')[1]) for par in parameters}

    lookback = int(par_dict['lb'])
    train_test_ratio = par_dict['tr']
    batch_size = 1

    df = load_data('train.csv')

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(df[['number_sold']])
    value_column = scaler.transform(df[['number_sold']])
    df['number_sold'] = value_column

    train_groups, test_groups = split_data(df, train_test_ratio)
    test_features, test_targets = create_rolling_windows(test_groups, lookback, 5)

    index = np.random.randint(0, len(test_features), 1)
    index = [2400]
    print(index)

    feature, target = torch.Tensor(test_features[index]), torch.Tensor(test_targets[index])
    # print(f'Feature {feature}')
    # print(target)
    predictions = []
    #  Predict 5 time steps
    for i in range(5):
        pred_value = model(feature)
        predictions.append(pred_value)
        feat_list = feature.tolist()[0]
        feat_list.pop(0)
        feat_list.append(pred_value.tolist())
        feature = torch.tensor([feat_list])

    fig, ax = plt.subplots()
    with torch.no_grad():
        ax.plot(range(lookback), test_features[index].tolist()[0], color='blue')
        ax.plot(range(lookback, lookback + 5), predictions, color='red')
        ax.plot(range(lookback, lookback + 5), test_targets[index].tolist()[0], color='green', linestyle='dotted')

    plt.show()


if __name__ == "__main__":
    #visualise_data()
    #visualize_training_groups()
    visualise_prediction()
