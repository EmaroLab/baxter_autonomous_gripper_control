#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package RawData_Interpolate
# This script generates sequences to fit a recurrent neural network, starting from separated files of Baxter and
# smartwatch raw data.
#
# For each sequence it interpolates Baxter data using the smartwatch timestamps.
# In this version, the final result does not consider timestamps as variable for the final dataset.

import os, os.path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

np.set_printoptions(suppress=True)

data_dir = os.path.dirname(os.getcwd()) + '/data'

new_cols_str = 'vx,vy,vz,ax,ay,az,'+\
                'angle_left_e0,angle_left_e1,angle_left_s0,angle_left_s1,angle_left_w0,angle_left_w1,angle_left_w2,'+\
                'velocity_left_e0,velocity_left_e1,velocity_left_s0,velocity_left_s1,velocity_left_w0,velocity_left_w1,velocity_left_w2,'+\
                'effort_left_e0,effort_left_e1,effort_left_s0,effort_left_s1,effort_left_w0,effort_left_w1,effort_left_w2,'+\
                'position_l_gripper'

n_seq = len(os.listdir(data_dir + '/raw_sequences'))//2

for seq_count in range(0, n_seq) :

    df_bax = pd.read_csv(data_dir + '/raw_sequences/baxter_raw_sequence'+str(seq_count))
    df_watch = pd.read_csv(data_dir +'/raw_sequences/smartwatch_raw_sequence'+str(seq_count))

    if (df_bax.empty or df_watch.empty) :
        continue

    bax_data = df_bax.values
    watch_data = df_watch.values

    #Dictionaries for columns index
    bax_cols = list(df_bax.columns.values)
    bax_cols = {k:v for v,k in enumerate(bax_cols)}
    watch_cols = list(df_watch.columns.values)
    watch_cols = {k:v for v,k in enumerate(watch_cols)}

    time_to_interpolate = watch_data[:,watch_cols['time']]

    train_msk = np.random.rand(bax_data.shape[0]) < 0.95
    bax_train = bax_data[train_msk, :]
    bax_test = bax_data[~train_msk, :]

    time_train = bax_train[:,bax_cols['time']]
    time_test = bax_test[:,bax_cols['time']]


    mse = []
    col_names = list(bax_cols.keys())
    idx = list(bax_cols.values())
    interpolated = np.empty((len(time_to_interpolate),len(bax_cols)-1))

    #Column-wise interpolation
    for col in range(0,(interpolated.shape[1]-1)) :
        f = interp1d(time_train, bax_train[:, col], kind='slinear', fill_value='extrapolate')
        interpolated[:,col] = f(time_to_interpolate)

        real_values = bax_test[:,col]
        err = np.power(real_values - f(time_test), 2)
        m = np.nanmean(err)
        v = np.nanstd(err)**2
        mse.append([(col_names[idx.index(col)]+' : '), ("%.15f" % m), ("%.15f" % v), \
                    ("%.5f %%" % (np.divide(v,m)*100))])

    #GRIPPER COL INTERPOLATION
    f_gripper = interp1d(bax_data[:,bax_cols['time']], bax_data[:, bax_cols['position_l_gripper']],\
                         kind='nearest', fill_value="extrapolate")

    #True stands for closed - False stands for opened
    interpolated[:,-1] = f_gripper(time_to_interpolate) <= 50

    #Uncomment to consider timestamps
    #interpolated[:,-1] = time_to_interpolate

    #Saving interpolation stats
    h ='Interpolation Stats: MSE mean and variance and (variance/mean)*100'
    np.savetxt(data_dir + '/interpolation_stats/stats_sequence'+str(seq_count)+'.txt', mse, \
                   delimiter='\t', header=h, fmt='%s')

    sequence = np.append(watch_data[:,1:-1], interpolated , axis=1)

    fmt_=''.join("%.3f,"*6) + ''.join("%.17f,"*14) + ''.join("%.3f,"*7) + "%d"

    np.savetxt(data_dir + '/dataset/sequence'+str(seq_count), sequence, \
                  delimiter=',', header=new_cols_str, comments='', fmt=fmt_)

print("Data successful interpolated!\nInterpolation statistics avalaible in [package]/data/interpolation_stats\n" +\
      "New merged sequences available in the [package]/data/dataset")