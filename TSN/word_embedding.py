# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:34:49 2019

@author: Shyam
"""
import os
import numpy as np
import torch
import string
import pandas as pd

class WordEmbedding:
    def __init__(self, series, embedding_file):
        self.embedding_file = embedding_file
        self.series = series
        self.dictionary_word_embeddings = {}
        #applying fill_dictionary to series, xx is redundant variable
        xx = self.series.apply(lambda x: self.fill_Dictionary(x))
        del(xx)
        self.load_embeddings(embedding_file)
    #Filling the dictionary_word_embeddings
    def fill_Dictionary(self, text):
        #strip the punctuations
        new_text = text.lower().translate(str.maketrans('', '', string.punctuation))
        all_words = new_text.split()
        for each in all_words:
            if each not in self.dictionary_word_embeddings:
                self.dictionary_word_embeddings[each] = None
                
    #load embeddings
    def load_embeddings(self,filename):
        with open(filename, 'r',encoding="utf8") as f:
            for line in f:
                vals = line.rstrip().split(' ')
                word = vals[0]
                if word in self.dictionary_word_embeddings:
                    self.dictionary_word_embeddings[word] = np.array([float(vals[i]) for i in range(1,len(vals))])


    def ConvertToEmbeddings(self, text):
        new_text = text.lower().translate(str.maketrans('', '', string.punctuation))
        all_words = new_text.split()
        count = 0
        for each in all_words:
            count += 1
            if each not in self.dictionary_word_embeddings or self.dictionary_word_embeddings[each] is None:
                embedding = self.dictionary_word_embeddings['the']
            else:
                embedding = self.dictionary_word_embeddings[each]
            if count == 1:
                embedding_average = embedding
            else:
                embedding_average = embedding_average + embedding
        embedding_average /= count
        return embedding_average

    def use_ConvertToEmbeddings(self, textInputs):
        count = 0
        for each in textInputs:
            count += 1
            embedding = self.ConvertToEmbeddings(each)
            embedding = embedding.reshape(1,-1)
            if count == 1:
                embedding_result = embedding
            else:
                embedding_result = np.concatenate((embedding_result,embedding), axis = 0)
        return embedding_result
    
    def numpy_to_tensor(self,np_array):
        return torch.tensor(np_array).float()
    
    #call function to make it easier
    def __call__(self, textInputs):
        result = self.use_ConvertToEmbeddings(textInputs)
        result = self.numpy_to_tensor(result)
        return result