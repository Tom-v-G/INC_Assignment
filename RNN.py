from model import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os

def mean_scores_dict():
    path = './scores'
    folder = os.fsencode(path)

    files = []

    mean_score = {}
    for file in os.listdir(folder):
        try:
            name = os.fsdecode(file)
            print(name)
            score_list = []
            with np.load(path + '/' + name, allow_pickle=True) as data:
                for item in data.files:
                    score_dict = data[item].item()

                for key, scores in score_dict.items():
                    score_list.append(np.mean(scores))
                    mean_score[(name, key)] = np.mean(scores)
                    print(np.mean(scores))
            final_score = np.mean(score_list)
            print(f'{name}: {final_score}')
        except:
            continue

    return mean_score
def main():
    score_dict = mean_scores_dict()


    #print(score_dict)
if __name__ == "__main__":
    main()
