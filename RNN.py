from model import RNN
import matplotlib.pyplot as plt

def visualise_data():
    test = RNN()
    data = test.load_data('~/Documents/INC/train.csv')
    fig, axs = plt.subplots(1, 7)
    for index, ax in enumerate(axs):
        for j in range(10):
            print( data.loc[ data['store'] == index & data['product'] == j, ['number_sold'] ])


def main():
    test = RNN()
    data = test.load_data('~/Documents/INC/train.csv')
    data.head()
    print(data)



if __name__ == "__main__":
    #main()
    visualise_data()