#import pytorch as pt
import numpy as np
import pandas as pd

class RNN:
    def __init__(self, nodes, inputs):
        self.state = 0
        self.weights = np.zeros()

    def load_data(self, path):
        """
        Loads data from a .csv file and stores it in a pandas dataframe
        :param path: string containing path to .csv file
        :return: None
        """

        raise NotImplementedError
    def train(self, data):
        """
        Trains the recurrent neural network on a dataset
        :param data:
        :return:
        """

        '''
        Initialisation:
        randomize weights.
        '''

        '''
        Forward pass:
        
        '''

        '''
        Backpropagation:
        unfold RNN in time, use classic backpropagation to calculate loss function for each timestep.
        Take average of weight gradients and update weights accordingly.
        '''
        raise NotImplementedError

    def evaluate(self, data):
        """
        Predicts values using trained RNN
        :param data:
        :return:
        """

        raise NotImplementedError

    def load_model(self, model_data, path):
        """
        Loads model from external dump file
        :param model_data:
        :param path:
        :return:
        """

        raise NotImplementedError

    def save_model(self, path):
        """
        Saves RNN model to disk
        :param path:
        :return:
        """


def main():
    print("hello world")


if __name__ == "__main__":
    main()