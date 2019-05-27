# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/27 21:56
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

import read_config
import read_data
import data_process
import models
import score_py3
import drawing
#%% Input Arguments
parser = argparse.ArgumentParser(description='Experiment3(Resnet): Train the model for diagnosing the heart disease by the ECG.')
parser.add_argument('-c', '--config', type=str, default='./Config/config.ini', metavar='str',
                    help="the path of configure file (default: './Config/config.ini')")
Args = parser.parse_args() # the Arguments
Args = read_config.read(Args) # read configure file
if Args.cuda:
    print('Using GPU.')
else:
    print('Using CPU.')
#%% Main Function
if __name__ == '__main__':
    # %% ########## Read Data ##########
    print('>>>>> Read Data')
    ECG_data, ECG_label = read_data.extract_data(read_data.read_data(Args))  # read data
    # %% ########## Data Processing ##########
    print('>>>>> Data Processing')
    ECG_data = data_process.split_wins(ECG_data, Args.win_size, Args.win_step)  # split windows
    ECG_label = data_process.label_from_0(ECG_label)  # label from 0
    # split data
    train_x, test_x, train_y, test_y = data_process.train_test_split(ECG_data, ECG_label,
                                                                     trainratio=Args.trainratio,
                                                                     random_state=0)
    # change to Tensor
    train_x, train_y = data_process.to_tensor(train_x, train_y)
    test_x, test_y = data_process.to_tensor(test_x, test_y)
    # %% ########## Load Data ##########
    print('>>>>> Load Data')
    # change to dataset
    # loader_train = data_process.dataloader(train_x, train_y, Args.batch_size, shuffle=True)
    loader_test = data_process.dataloader(test_x, test_y, Args.batch_size, shuffle=True)
    del ECG_data, ECG_label
    # %% ########## Create model ##########
    print('>>>>> Create model')
    cnn_bilstm = models.CNN_BiLSTM().cuda() if Args.cuda else models.CNN_BiLSTM()
    # optimizer
    optimizer = torch.optim.Adam(cnn_bilstm.parameters(), lr=Args.learn_rate)
    # loss function
    loss_func = nn.CrossEntropyLoss()
    # evaluate
    Accuracy = []
    F1 = []
    # %% ########## Training ##########
    if Args.show_plot:
        fig = plt.figure(1)
        plt.ion()
    print('>>>>> Start Training')
    for epoch in range(Args.epoch):
        # load data
        loader_train = data_process.dataloader(train_x, train_y, Args.batch_size, shuffle=True)
        break
        ##### Train #####
        for step, (x, y) in enumerate(loader_train):  # input batch data from train loader
            ##### learning #####
            output = []
            cnn_bilstm.train()
            for i in x:
                input = i.cuda() if Args.cuda else i
                output.append(cnn_bilstm(input))
            output = torch.cat(output)
            y = torch.FloatTensor(y).type(torch.LongTensor).cuda() if Args.cuda else torch.FloatTensor(y).type(torch.LongTensor)
            loss = loss_func(output, y)  # loss
            # optimizer.zero_grad()  # clear gradients for next train
            # loss.backward()  # backpropagation, compute gradients
            # optimizer.step()  # apply gradients
            # if step % 1 == 0:
            #     if PARM.ifGPU:
            #         pred = torch.max(output, 1)[1].cuda().data.squeeze()
            #     else:
            #         pred = torch.max(output, 1)[1].data.squeeze()
            #     accuracy = float((pred == y).sum()) / float(y.size(0))
            #     #            F1 = score_s(pred, y)
            #     print('Epoch: %s |step: %s | train loss: %.2f | accuracy: %.2f '
            #           % (epoch, step, loss.data, accuracy))
            #     del x, y, input, output
            #     if PARM.ifGPU: torch.cuda.empty_cache()
            #
            # if Args.cuda:
            #     x = x.cuda()
            #     y = y.cuda()
            # cnn_bilstm.train()
            # output = cnn_bilstm(x)
            # loss = loss_func(output, y)  # get loss
            # optimizer.zero_grad()  # clear gradients for backward
            # loss.backward()  # backpropagation, compute gradients
            # optimizer.step()  # apply gradients

