from model import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualise_data():
    test = RNN()
    data = test.load_data('~/Documents/INC/train.csv')
    fig, axs = plt.subplots(1, 7)
    for index, ax in enumerate(axs):
        for j in range(10):
            print( data.loc[ data['store'] == index & data['product'] == j, ['number_sold'] ])


def main():
    window_size = 10
    #test = RNN(1)
    data = load_data('train.csv')
    print(data)
    #df = data.groupby(["store", "product"])
    train_groups, test_groups = split_data(data, 0.8)
    #print(train_groups[(0, 2)])
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #tr = scaler.fit_transform(train_groups[(0, 2)])
    #print(tr)
    #print(scaler.inverse_transform(tr))
    X, y = [], []
    print(len(train_groups[(0, 0)]) )
    for i in range(window_size + 1, len(train_groups[(0, 0)]) - 5):
        X.append(train_groups[(0, 0)][i - window_size + 1: i + 1].astype('float32'))
        y.append(train_groups[(0, 0)].loc[:, 'number_sold'].iloc[i + 1: i + 6].values.astype('float32'))
    print(X)
    print(y)




if __name__ == "__main__":
    main()
