#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package PCA_SplinedData
# This script analyzes the dataset with a Principal Component Analysis approach.
# It provide a visualization for results and a score for each variable.
#
# In this version sequences are cutted to consider only the grasping phase.

import os, os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing as prep

dataset_dir = os.path.dirname(os.getcwd()) + '/data/dataset'
n_seq = len(os.listdir(dataset_dir))

seq_len = 25

#Collect all data (all sequences)
for seq_count in range(0, n_seq) :
    seq = pd.read_csv(dataset_dir + '/sequence'+str(seq_count))

    #Initialize seq container and columuns
    if (seq_count==0) :
        cols = list(seq.columns.values)
        all_seq = pd.DataFrame(data=[], columns=cols)

    first_one = seq.loc[:,'position_l_gripper'].idxmax()
    #Cutting the sequence
    seq = seq.loc[:(first_one + seq_len),:]
    all_seq = all_seq.append(seq, ignore_index=True, sort=False)


# PRINCIPAL COMPONENT ANALYSIS

scaled_data = prep.scale(all_seq.loc[:,cols[0:-1]])
pca = PCA()
pca.fit(scaled_data)

variation_percentage = np.round(pca.explained_variance_ratio_*100, decimals=1)
#1-dim array for the single PCi : pca.components_[i]
loading_scores = pd.DataFrame(pca.components_.T, index=cols[0:-1])
loading_scores = abs(loading_scores)

score = loading_scores * variation_percentage
mean_score = score.mean(axis=1)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid(color='tab:gray', linestyle='-', linewidth=0.5)
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()