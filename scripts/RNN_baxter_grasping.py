#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package RNN_baxter_grasping
# This script preprocesses sequence data to fit a recurrent neural network and then it estimates and
# evaluate a Keras sequential model.
#
# It saves trainig logs for a tensorboard visualization and it saves the estimated model.

import pandas as pd
import numpy as np
import math
import os, os.path
from time import time

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dense, Dropout, LSTM, Embedding
from sklearn.model_selection import train_test_split

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

pkg_dir = os.path.dirname(os.getcwd())
dataset_dir = pkg_dir + '/data/dataset'

n_seq = len(os.listdir(dataset_dir))

seq_len = 25

#%%
#Collecting data for ZSCORE normalization
cols = []
for seq_count in range(0, n_seq) :

    seq = pd.read_csv(dataset_dir + '/sequence'+str(seq_count))
    #Initialize seq container and columuns
    if (seq_count==0) :
        cols = list(seq.columns.values)
        all_seq = pd.DataFrame(data=[], columns=cols)

    first_one = seq.loc[:,'position_l_gripper'].idxmax()
    seq = seq.loc[:(first_one + seq_len),:]
    all_seq = all_seq.append(seq, ignore_index=True, sort=False)

mean_all_seq = all_seq.iloc[:,:-1].mean()
std_all_seq = all_seq.iloc[:,:-1].std()
zscore_data = pd.DataFrame([mean_all_seq, std_all_seq], index=['mean', 'std'])
#Saving zscore mean and std to use the model in real time
np.savetxt(pkg_dir + '/model/mean_std_zscore', zscore_data.values, \
                  delimiter=',', comments='')

#%%
#PREPROCESSING & SERIES GENERATION
dataset = []
targets = []
post_stride = 1
for seq_count in range(0, n_seq) :

    df = pd.read_csv(dataset_dir + '/sequence'+str(seq_count), float_precision='round_trip')
    # DATASET CUTTING
    first_one = df.loc[:,'position_l_gripper'].idxmax()

    df = df.loc[:(first_one + seq_len),:]
    values = df.values
    seq = values[:,:-1]
    targ = values[:,-1]
    targ = targ[:, np.newaxis]

    #ZSCORE normalization
    seq = seq - zscore_data.loc['mean',cols[:-1]].values
    seq = seq/zscore_data.loc['std',cols[:-1]].values

    #SERIES GENERATION
    subseq_num = math.ceil(((seq_len*2)-seq_len+1)/post_stride)
    pre_stride = math.ceil(((first_one)-seq_len+1)/subseq_num)

    pre_series = TimeseriesGenerator(seq[:first_one,:], targ[:first_one], length=seq_len, stride=pre_stride,
                                      batch_size=subseq_num)
    post_series = TimeseriesGenerator(seq[first_one-seq_len:,:], targ[first_one-seq_len:], length=seq_len, stride=post_stride,
                                      batch_size=subseq_num)

    x, y = pre_series[0]
    dataset.extend(x.tolist())
    targets.extend(y.tolist())

    x, y = post_series[0]
    dataset.extend(x.tolist())
    targets.extend(y.tolist())

dataset = np.array(dataset)
targets = np.array(targets)

x_train,x_val,y_train,y_val = train_test_split(dataset, targets, test_size=0.20)

#%%
#Model estimation and evaluation
tbCallBack = keras.callbacks.TensorBoard(log_dir=pkg_dir+"/model_logs/{}".format(time()), write_graph=True)

model = keras.models.Sequential()

model.add(keras.layers.LSTM(1, input_shape=(seq_len,dataset.shape[2])))
model.add(keras.layers.Dense(1, activation='sigmoid'))

sgd=tf.keras.optimizers.SGD(lr=0.9, momentum=0.7, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
scores = model.fit(x_train, y_train, epochs=20, verbose=1, callbacks=[tbCallBack])

scores = model.evaluate(x_val, y_val, verbose=1)
print(str(model.metrics_names) + " " + str(scores))

model.save(pkg_dir + '/model/tmp_models/my_model' + str(seq_len) + '-' + str(int(scores[1]*100)) + '.h5')

#%%
original_res = []
res=[]
ndarr = np.ndarray((1, seq_len, 27))
for i in dataset :
    ndarr[0] = i
    r = model.predict(ndarr)
    res.append([np.array(1) if r[0][0] > 0.7 else np.array(0)])
    original_res.append(r[0][0])

import matplotlib.pyplot as plt
plt.figure(figsize=(200,10))
plt.plot(original_res)
plt.plot(np.linspace(0,1800,1000),np.full((1000,),0.5))
plt.xlim(0)
plt.show()