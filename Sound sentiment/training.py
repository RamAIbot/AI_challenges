# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:49:14 2021

@author: Admin
"""

import librosa
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm

def extract_feature(file_name):
    
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
        result = np.hstack((result,mfccs))
        stft = np.abs(librosa.stft(X))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft,sr=sample_rate).T,axis=0)
        result = np.hstack((result,chroma))
        mel = np.mean(librosa.feature.melspectrogram(X,sr=sample_rate).T,axis=0)
        result = np.hstack((result,mel))
    return result
    #180 dimensional row (40+12+128)


def load_data(file_path,label_path):
    x,y=[],[]
    data = pd.read_csv(os.path.join(label_path))
    for file in tqdm(os.listdir(file_path)):
        file_name = os.path.join(file_path + file)
        label_name = file.split('.')[0]
        features = extract_feature(file_name)
        x.append(features)
        label = data.loc[data['wav_id']==int(label_name),'label']
        y.append(label)
        
    return np.array(x),np.array(y)

        
    
train_path = './train/'
train_file = 'train (1).csv'
X_train,Y_train = load_data(train_path,train_file)
print(X_train.shape)
print(Y_train.shape)