# -*- coding: utf-8 -*-
"""
Created on Tue May 14 23:11:32 2019

@author: Shyam
Lets split the QA files into train and test data
We will split data into 80-20%.

Since we are doing baseline method, we will not downsample
Arguments - downsample = True / False

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def F_downsample(downsample):
    pass

"""
File reading, where we will take question - answer file (individual segment), and SoftmaxIndex file
1. We will filter out data which are not in softmaxindex file using pandas inbuilt isin function
2. We will split by y variable and Qtype.
"""
def read_file(filename):
    return pd.read_csv(filename)

def Generate_answer_list(df):
    #df = read_file(filename)
    answer_list = np.array(df['answer'])
    return answer_list

def Train_Test_Split(df, answer_list):
    df_subset = df[df['answer'].isin(answer_list)]
    train, test = train_test_split(df_subset, test_size=0.2)
    return train, test

"""main program starts below"""
#generating answer list
filename = 'SoftmaxIndex.csv'
df_SoftmaxIndex = read_file(filename)
answer_list = Generate_answer_list(df_SoftmaxIndex)

#generating train test split
filename = 'QA_Individual_segments.csv'
df_QA_segments = read_file(filename)
trainData, testData = Train_Test_Split(df_QA_segments, answer_list)

#writing to files
trainData.to_csv('train_QA.csv', index = False)
testData.to_csv('test_QA.csv', index = False)

