import numpy as np
import pandas as pd

from model import *
import torch.utils.data as data




if __name__ == "__main__":
    path = 'train.csv' #update if train.csv is in a different directory
    window_size = 10 #change to different values for
    df = load_data(path)
    train_groups, test_groups = split_data(df, 0.67)

    print(create_window(train_groups[(0, 0)], 20, 10))
