# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:46:27 2019

@author: Shyam
"""

import os
os.chdir('C:\stuff\Studies\Spring 19\Independent Study\Data\Epic Kitchens Dataset\QA Generation')

"""
Alright let's start QA generation
"""

import pandas as pd
import numpy as np

df_train_action = pd.read_csv('../EPIC_train_action_labels.csv')

df_train_action.columns

"""
Let's now form questions about actions on noun classes:
Eg. What did I do with the door?

For any video, we will subset by the noun, and list down the actions. That will be our answer.

"""
#string = 'kalou:nattu'
#Removing colon from compound nouns for realism
def function_colon_rem(string):
    if ':' in string:
        string = string.split(':')
        string = string[1] + ' '+ string[0]
    return string
        
df_train_action['noun_clean'] = df_train_action['noun']
df_train_action['noun_clean'] = df_train_action['noun_clean'].apply(function_colon_rem)

"""

We will use noun_clean column to form questions, see occurances of nouns and answer accordingly.

"""
df_train_action['question_about_action'] = "What did he do with the " + df_train_action['noun_clean']

df_video_questions = df_train_action.groupby(['video_id','question_about_action','noun'], as_index = False).size().reset_index()
df_video_questions = df_video_questions[['video_id','question_about_action','noun']]
df_video_questions['answer'] = ''
df_video_questions['time'] = ''
df_video_questions['frames'] = ''


final_answers_array = []
final_time_array = []
final_frames_array = []

for i in range(df_video_questions.shape[0]):
    video_id = df_video_questions.iloc[i,0]
    noun = df_video_questions.iloc[i,2]
    df_train_action_subset = df_train_action[(df_train_action['video_id'] == video_id) & (df_train_action['noun'] == noun)]
    all_actions = np.array(df_train_action_subset['verb'])
    #start and end time
    all_start_times = np.array(df_train_action_subset['start_timestamp'])
    all_end_times = np.array(df_train_action_subset['stop_timestamp'])  
    
    #start and end frames
    all_start_frames = np.array(df_train_action_subset['start_frame'])
    all_end_frames = np.array(df_train_action_subset['stop_frame'])  
    
    #Setting up final strings    
    final_answer_string = ''
    final_time_string = ''
    final_frames_string = ''

    for i in range(len(all_actions)):
        final_answer_string += ',' + all_actions[i]
        final_time_string += ',' + all_start_times[i] + '-' + all_end_times[i]
        final_frames_string += ',' + str(all_start_frames[i]) + '-' + str(all_end_frames[i])
        
    final_answer_string = final_answer_string[1:]
    final_time_string = final_time_string[1:]
    final_frames_string = final_frames_string[1:]
    
    final_answers_array.append(final_answer_string)
    final_time_array.append(final_time_string)
    final_frames_array.append(final_frames_string)
    
df_video_questions['answer'] = np.array(final_answers_array)   
df_video_questions['time'] = np.array(final_time_array)   
df_video_questions['frames'] = np.array(final_frames_array)

df_video_questions.to_csv('QA_actions.csv', index = False)        

"""
We have till now set some base for questions about actions / verbs.
Now we can form questions about nouns.
Then later we can address compound noun questions (like put pizza on the pan, where pizza and pan are nouns).

In question about nouns (what all did he turn-on?)
(What all did he ignite, begin, activate etc. So many are there.)
As of now, I am not grouping them.

Later on, I will make a list of cleaned nouns and cleaned verbs, which have similar meaning. 
This will be used for action recognition pipeline.

"""



df_train_action['question_about_nouns'] = "What all does he " + df_train_action['verb']

df_video_questions = df_train_action.groupby(['video_id','question_about_nouns','verb'], as_index = False).size().reset_index()
df_video_questions = df_video_questions[['video_id','question_about_nouns','verb']]
df_video_questions['answer'] = ''
df_video_questions['time'] = ''
df_video_questions['frames'] = ''


final_answers_array = []
final_time_array = []
final_frames_array = []

for i in range(df_video_questions.shape[0]):
    video_id = df_video_questions.iloc[i,0]
    verb = df_video_questions.iloc[i,2]
    df_train_action_subset = df_train_action[(df_train_action['video_id'] == video_id) & (df_train_action['verb'] == verb)]
    all_nouns = np.array(df_train_action_subset['noun_clean'])
    #start and end time
    all_start_times = np.array(df_train_action_subset['start_timestamp'])
    all_end_times = np.array(df_train_action_subset['stop_timestamp'])  
    
    #start and end frames
    all_start_frames = np.array(df_train_action_subset['start_frame'])
    all_end_frames = np.array(df_train_action_subset['stop_frame'])  
    
    #Setting up final strings    
    final_answer_string = ''
    final_time_string = ''
    final_frames_string = ''

    for i in range(len(all_nouns)):
        final_answer_string += ',' + all_nouns[i]
        final_time_string += ',' + all_start_times[i] + '-' + all_end_times[i]
        final_frames_string += ',' + str(all_start_frames[i]) + '-' + str(all_end_frames[i])
        
    final_answer_string = final_answer_string[1:]
    final_time_string = final_time_string[1:]
    final_frames_string = final_frames_string[1:]
    
    final_answers_array.append(final_answer_string)
    final_time_array.append(final_time_string)
    final_frames_array.append(final_frames_string)
    
df_video_questions['answer'] = np.array(final_answers_array)   
df_video_questions['time'] = np.array(final_time_array)   
df_video_questions['frames'] = np.array(final_frames_array)

df_video_questions.to_csv('QA_nouns.csv', index = False)        

"""
Let's now try interaction questions
"""

