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
parser.add_argument('-tr', '--trainratio', type=float, default=0.9, metavar='float',
                    help="proportion of training sets. (default: 0.9)")
parser.add_argument('-w', '--win-size', type=int, default=4000, metavar='int',
                    help="length of the ecg data which be cutted out. (default: 3000)")
parser.add_argument('-s', '--win-step', type=int, default=2000, metavar='int',
                    help="length of the ecg data which be cutted out. (default: 3000)")
parser.add_argument('-b', '--batch-size', type=int, default=100, metavar='int',
                    help="the batch size for training")
parser.add_argument('-e', '--epoch', type=int, default=50, metavar='int',
                    help="epoch for training. (default: 50)")
parser.add_argument('-lr', '--learn-rate', type=float, default=0.001, metavar='float',
                    help="the learning rate. (default: 0.001)")
parser.add_argument('-c', '--cuda', type=bool, default=True, metavar='bool',
                    help="enables CUDA training. (default: True)")
parser.add_argument('-s', '--show', type=bool, default=True, metavar='bool',
                    help="show the result by matplotlib. (default: True)")
parser.add_argument('-dp', '--data-path', type=str, default=r"D:\My DataBases\ECG_data\TrainingSet1", metavar='str',
                    help="the path of the data. (default: 'D:\My DataBases\ECG_data\TrainingSet1')")
parser.add_argument('-lp', '--label_path', type=str, default=r'D:\My DataBases\ECG_data\TrainingSet1\REFERENCE.csv', metavar='str',
                    help="the path of the label file. (default: 'D:\My DataBases\ECG_data\TrainingSet1\REFERENCE.csv')")

Args = parser.parse_args() # the Arguments
