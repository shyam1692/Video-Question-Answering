# -*- coding: utf-8 -*-
"""
Created on Mon May 13 23:11:00 2019

@author: Shyam

Framewise Question answers generation.
Takes each files, subsets by video column by videos, and answer column such that answer is not blank.
Output is a filewise segment wise video id, question, answer, time and frames

Arguments - Video list, Question type list,file list
"""
import os
os.chdir('C:\stuff\Studies\Spring 19\Independent Study\Data\Epic Kitchens Dataset\QA Generation')
import pandas as pd
import numpy as np

def read_file(filename, Qtype, VideoList):
    df = pd.read_csv(filename)
    df = df[df['answer'] != '']
    df = df[df['video_id'].isin(VideoList)]
    df.dropna(inplace = True)
    #iterating for each row below and from a row generating QAs to be appended.
    df_output = pd.DataFrame(columns = ['video_id','question','answer','time','frames', 'Qtype'])
    for i in range(df.shape[0]):
        frames = np.array(df['frames'])[i].split(',')
        frames = np.array(frames)
        df_row = pd.DataFrame(frames, columns = ['frames'])
        times = np.array(df['time'])[i].split(',')
        times = np.array(times)
        answers = np.array(df['answer'])[i].split(',')
        answers = np.array(answers)        
        df_row['time'] = times
        df_row['answer'] = answers
        df_row['question'] = np.array(df['question'])[i]
        df_row['Qtype'] = Qtype
        df_row['video_id'] = np.array(df['video_id'])[i]
        df_output = pd.concat((df_output, df_row[['video_id','question','answer','time','frames', 'Qtype']]), axis = 0)
    #for loop ends
    return df_output

#we have list of files, and their corresponding Qtypes.
fileList = ['QA_actions.csv','QA_dontknow_actions.csv','QA_dontknow_nouns.csv', 
            'QA_interaction_1.csv','QA_interaction_1_dontknow.csv','QA_interaction_2.csv','QA_interaction_2_dontknow.csv','QA_nouns.csv']

QtypeList = ['action', 'action','noun','interaction1','interaction1','interaction2','interaction2','noun']

VideoList = ['P01_02','P01_03','P01_04','P01_06','P01_07','P02_01','P02_04','P03_02','P03_09']

df_QA_segments = pd.DataFrame(columns = ['video_id','question','answer','time','frames', 'Qtype'])
#reading each file, and updating pandas dataframe
for i, filename in enumerate(fileList):
    print('File being done is ' + filename)
    Qtype = QtypeList[i]
    df_output = read_file(filename, Qtype, VideoList)
    df_QA_segments = pd.concat((df_QA_segments, df_output), axis = 0)

"""Writing result to final file"""
df_QA_segments.to_csv('QA_Individual_segments.csv', index = False) 