#
# Data visualisation
#a

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import RNN

def visualise_data():
    test = RNN()
    df = test.load_data('train.csv')

    gb = df.groupby(["store", "product"])
    groups = {x: gb.get_group(x) for x in gb.groups}

    fig, axs = plt.subplots(7, 10, figsize=(40, 35))
    for index, store_axs in enumerate(axs):
        for index2, product_ax in enumerate(store_axs):
            X = groups[(index, index2)]
            X = X.drop(["Date", "store", "product"], axis=1).values
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


if __name__ == "__main__":
    visualise_data()
