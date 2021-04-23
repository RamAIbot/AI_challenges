# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:29:29 2021

@author: Admin
"""

import librosa
import soundfile
import numpy as np
import matplotlib.pyplot as plt

file_name = './train/0.wav' #=> negative
file_name = './train/17.wav' #=> neutral
file_name = './train/18.wav' #=> positive

with soundfile.SoundFile(file_name) as sound_file:
    X = sound_file.read(dtype="float32")
    sample_rate = sound_file.samplerate
    
#    samples =  np.arange(0, 10, 0.1);
#    signal = np.sin(samples) 
    
    print(X.shape)
    print(X[0:10])
    print(sample_rate)
    plt.plot(X)
    plt.show()
    
    #return 40x103 (103 => for each freq 40 coeff)
    mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
    print(mfccs.shape)
    print(mfccs[0:10])
    plt.plot(mfccs)
    plt.show()
    
#    new_mfcc = librosa.feature.mfcc(y=signal,sr=0.05,n_mfcc=40)
#    plt.plot(signal)
#    plt.show()
#    
#    plt.plot(new_mfcc)
#    plt.show()
    stft = np.abs(librosa.stft(X))
    #return 12x103
    chroma = np.mean(librosa.feature.chroma_stft(S=stft,sr=sample_rate).T,axis=0)
    print(chroma.shape)
    plt.plot(chroma)
    plt.show()
    
    #return (128, 103)
    mel = np.mean(librosa.feature.melspectrogram(X,sr=sample_rate).T,axis=0)
    print(mel.shape)
    plt.plot(mel)
    plt.show()