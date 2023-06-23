Time Series Forecasting with a RNN.
Created by:
Lisa Schouten, s3162915
Tom v. Gelooven, s1853686


For this assignment we have created a RNN with the ability to predict time series of the 'train.csv' datafile.

Data is preprocessed the following way:
 1. The 'Date' column is removed from the datafile.
 2. The 'products_sold' column is scaled to values between 0 and 1 (using sklearn MinMaxScalar()).
 3. The data is grouped per store and product and stored in a dictionary of dataframes.
 4. Each dataframe is split into a training and a testing dataframe (for the final version of our RNN we used 67% of the
    data as training data and the rest for testing).
 4a. Depending on if the model is being trained or being tested the corresponding dataset is used.
 5. Each dataset is divided into feature 'rolling windows' and target 'rolling windows'.
    The first n_lookback entries are the data given to the model, stored in a feature tensor.
    The next n_target entries are the targets to be predicted when training / testing, stored in a target tensor.
 6. All rolling windows are loaded together into a torch.Dataloader.

To train the model we used batch gradient descent with MAPE as the loss function. We used a batch size of 64.

The architecture we chose for was a layered LSTM. To initialise this model we provide it with the amount of hidden nodes
we wish to have in the network, together with how many layers of LSMT modules we want to have.

Many different configurations, with different lookback windows, amount of hidden nodes, learning rates,
training times (in epochs), and amount of LSTM layers were tried. We got the best result with the following:

- A single LSTM module
- A learning rate of 0.0001
- 128 hidden nodes
- 500 epochs of training
- a lookback window of length 10

Note: if train.py does not run first try replacing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') with
device = torch.device('cpu')

Kind regards,

Lisa and Tom
:)




