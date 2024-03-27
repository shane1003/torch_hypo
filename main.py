import os
import time
import yaml
import math
import pandas as pd
import numpy as np

from data.excel2pd import data_load
from data.preprocessing import data_preprocessing
from data.train_test_split import data2Window, X_Y2train_test, X_Y_from_data

import torch
import torch.nn as nn
import torch.optim as optim

def main():
    f = open("Settings.yaml", 'r')
    setup = yaml.load(f, Loader=yaml.FullLoader)

    data = data_load(11)
    scaler, data = data_preprocessing(data, setup['data_options'])

    X, Y = X_Y_from_data(data)

    X_train, Y_train, X_test, Y_test = X_Y2train_test(X, Y,
                                                    setup['input_window'],
                                                    setup['output_window'],
                                                    setup['stride'])

if __name__ == '__main__':
    main()
