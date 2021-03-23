# -*- coding: utf-8 -*-
"""
Created on Fri May 15 02:18:43 2020

@author: prach
"""
#import libraries
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.output
import librosa.display
from scipy.spatial import distance

#load audios : original and cover song
y, sr = librosa.load('countingstars_original.mp3')
y1 , sr1 = librosa.load('countingstars_cover.mp3')

# Compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))
S_full_1, phase_1 = librosa.magphase(librosa.stft(y1))

''' ORIGINAL SONG '''
#plot slice 
idx = slice(*librosa.time_to_frames([1, 11], sr=sr))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max), y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.tight_layout()

S_filter = librosa.decompose.nn_filter(S_full,aggregate=np.median, metric='cosine',width=int(librosa.time_to_frames(2, sr=sr)))
S_filter = np.minimum(S_full, S_filter)
# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
# Note: the margins need not be equal for foreground and background separation
margin_i, margin_v = 2, 10
power = 2
mask_i = librosa.util.softmask(S_filter,margin_i * (S_full - S_filter),power=power)
mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter,power=power)
# Once we have the masks, simply multiply them with the input spectrum
# to separate the components
S_foreground = mask_v * S_full
S_background = mask_i * S_full
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max), y_axis='log', sr=sr)
plt.title('Full spectrum :original')
plt.colorbar()
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),y_axis='log', sr=sr)
plt.title('Background : original')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground : original')
plt.colorbar()
plt.tight_layout()
plt.show()

type(S_foreground)
# import numpy as np
# import sounddevice as sd
# fs = 44100
# data = np.random.uniform(-1, 1, fs)
# sd.play(S_foreground, fs)
# from scipy.io.wavfile import write
# samplerate = 44100; fs = 100
# t = np.linspace(0., 1., samplerate)
# amplitude = np.iinfo(np.int16).max
# data = amplitude * np.sin(2. * np.pi * fs * t)
# write("example.wav", samplerate, data)

# =============================================================================
# import numpy as np
# import sounddevice as sd
# import time
# sps = 44100
# freq_hz = 440.0
# duration_s = 5.0
# atten = 0.3
# # NumpPy magic to calculate the waveform
# each_sample_number = S_foreground.arange(duration_s * sps)
# waveform = np.sin(2 * np.pi * each_sample_number * freq_hz / sps)
# waveform_quiet = waveform * atten
# 
# # Play the waveform out the speakers
# sd.play(waveform_quiet, sps)
# time.sleep(duration_s)
# sd.stop()
# =============================================================================


''' COVER SONG'''
#parameters: S_full_1, phase_1 , y1 , sr1

#plot slice 
idx_1 = slice(*librosa.time_to_frames([1, 11], sr=sr1))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_full_1[:, idx_1], ref=np.max), y_axis='log', x_axis='time', sr=sr1)
plt.colorbar()
plt.tight_layout()

S_filter_1 = librosa.decompose.nn_filter(S_full_1,aggregate=np.median, metric='cosine',width=int(librosa.time_to_frames(2, sr=sr1)))
S_filter_1 = np.minimum(S_full_1, S_filter_1)
# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
# Note: the margins need not be equal for foreground and background separation
margin_i_1, margin_v_1 = 2, 10
power_1 = 2
mask_i_1 = librosa.util.softmask(S_filter_1,margin_i_1 * (S_full_1 - S_filter_1),power=power_1)
mask_v_1 = librosa.util.softmask(S_full_1 - S_filter_1, margin_v_1 * S_filter_1,power=power_1)
# Once we have the masks, simply multiply them with the input spectrum
# to separate the components
S_foreground_1 = mask_v_1 * S_full_1
S_background_1 = mask_i_1 * S_full_1 
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full_1[:, idx_1], ref=np.max), y_axis='log', sr=sr1)
plt.title('Full spectrum : cover')
plt.colorbar()
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background_1[:, idx_1], ref=np.max),y_axis='log', sr=sr1)
plt.title('Background : cover')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground_1[:, idx_1], ref=np.max),y_axis='log', x_axis='time', sr=sr1)
plt.title('Foreground : cover')
plt.colorbar()
plt.tight_layout()
plt.show()

type(S_foreground_1)


''' COMPARISON '''
org= S_foreground
cov= S_foreground_1
print("Dimensions of original song vocals:",org.shape)
print("Dimensions of cover song vocals:",cov.shape)
type(org)
#broadcasting
#np.dot(cov,org).shape
# dist = np.linalg.norm(org-cov)

#flatenning
org=org.flatten()
o=org.shape

cov=cov.flatten()
c=cov.shape

#reshape
np.reshape(org, (o[0], 1))
org.shape

np.reshape(cov, (c[0], 1))
cov.shape

diff= org.shape[0] - cov.shape[0]
cov.shape[0] + diff
# for i in range(0,11361100):
#     np.append(cov, 0)
# cov.shape
# for i in range(0,diff):
#     np.append(cov,0)
# cov.shape

# convert numpy array to list
org1=list(org)
cov1=list(cov)

#reshaping
for i in range(diff):
    cov1.append(0)

#distance calculation
dis=distance.euclidean(org1, cov1)
print("Euclidean distance between arrays of original song and cover song =",dis)

#using dynamic time wrapping
import dtw
#from dtaidistance import dtw
type(org)
distance=dtw.distance(org1,cov1)
dtw.Distance(org1,cov1)
from dtw import *
from fastdtw import fastdtw
distance = fastdtw(org1, cov1)[0]
print("Distance between original and cover song using dtw=",distance)

#Similarity % between vocals
res = len(set(org1) & set(cov1)) / float(len(set(org1) | set(cov1))) * 100

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
y = np.array([[2,2], [3,3], [4,4]])
distance, path = fastdtw(org1, cov1, dist=euclidean)
print(distance)
