# -*- coding: utf-8 -*-
"""
Created on Tue May 14 00:09:47 2019

@author: Shyam

This file will load all QA
Creating answers softmax index
"""

import os
os.chdir('C:\stuff\Studies\Spring 19\Independent Study\Data\Epic Kitchens Dataset\QA Generation')
import pandas as pd
import numpy as np

"""creating a dictionary and storing answers, and corresponding index"""

def read_file(filename):
    df = pd.read_csv(filename)
    df_softmax_index = pd.DataFrame(df.groupby(['answer'], as_index = False).size().reset_index())
    df_softmax_index = df_softmax_index.rename(columns={0:'Count'})
    df_softmax_index['softmax_index'] = np.array(range(df_softmax_index.shape[0]))
    return df_softmax_index

filename = 'QA_Individual_segments.csv'
df_softmax_index = read_file(filename)
df_softmax_index.to_csv('SoftmaxIndex.csv', index = False) 