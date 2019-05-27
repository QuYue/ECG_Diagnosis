# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/27 14:23
@Author  : QuYue
@File    : ECG_train.py
@Software: PyCharm
Introduction: Train the model for diagnosing the heart disease by the ECG.
"""
#%% Import Packages
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import read_data
import data_process
import models
import score_py3
import drawing
#%% Input Arguments
parser = argparse.ArgumentParser(description='Experiment1(CNN): Train the model for diagnosing the heart disease by the ECG.')
parser.add_argument('-n', '--datanum', type=int, default=2000, metavar='int',
                    help="the number of data. (default: 2000)")

Args = parser.parse_args() # the Arguments
