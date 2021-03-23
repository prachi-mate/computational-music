# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:56:36 2020

@author: prach
"""

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
y1 , sr1 = librosa.load('vandematramanuja.wav')
y2 , sr2 = librosa.load('vandematramlata.wav')
y3 , sr3 = librosa.load('vandematramshreya.wav')

# Compute the spectrogram magnitude and phase
S_full_1, phase_1 = librosa.magphase(librosa.stft(y1))
S_full_2, phase_2 = librosa.magphase(librosa.stft(y2))
S_full_3, phase_3 = librosa.magphase(librosa.stft(y3))

'''SONG 1: vandematramanuja'''
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
plt.title('Full spectrum : vandematram_anuja')
plt.colorbar()
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background_1[:, idx_1], ref=np.max),y_axis='log', sr=sr1)
plt.title('Background : vandematram_anuja')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground_1[:, idx_1], ref=np.max),y_axis='log', x_axis='time', sr=sr1)
plt.title('Foreground : vandematram_anuja')
plt.colorbar()
plt.tight_layout()
plt.show()



'''SONG 2: vandematramlata'''
#parameters: S_full_1, phase_1 , y1 , sr1
#plot slice 
idx_2 = slice(*librosa.time_to_frames([1, 11], sr=sr2))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_full_2[:, idx_2], ref=np.max), y_axis='log', x_axis='time', sr=sr2)
plt.colorbar()
plt.tight_layout()

S_filter_2 = librosa.decompose.nn_filter(S_full_2,aggregate=np.median, metric='cosine',width=int(librosa.time_to_frames(2, sr=sr2)))
S_filter_2 = np.minimum(S_full_2, S_filter_2)
# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
# Note: the margins need not be equal for foreground and background separation
margin_i_2, margin_v_2 = 2, 10
power_2 = 2
mask_i_2 = librosa.util.softmask(S_filter_2,margin_i_2 * (S_full_2 - S_filter_2),power=power_2)
mask_v_2 = librosa.util.softmask(S_full_2 - S_filter_2, margin_v_2 * S_filter_2,power=power_2)
# Once we have the masks, simply multiply them with the input spectrum
# to separate the components
S_foreground_2 = mask_v_2 * S_full_2
S_background_2 = mask_i_2 * S_full_2 
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full_2[:, idx_2], ref=np.max), y_axis='log', sr=sr2)
plt.title('Full spectrum : vandematram_lata')
plt.colorbar()
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background_2[:, idx_2], ref=np.max),y_axis='log', sr=sr2)
plt.title('Background : vandematram_lata')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground_2[:, idx_2], ref=np.max),y_axis='log', x_axis='time', sr=sr2)
plt.title('Foreground : vandematram_lata')
plt.colorbar()
plt.tight_layout()
plt.show()


'''SONG 3: vandematramshreya'''
#parameters: S_full_1, phase_1 , y1 , sr1
#plot slice 
idx_3 = slice(*librosa.time_to_frames([1, 11], sr=sr3))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_full_3[:, idx_3], ref=np.max), y_axis='log', x_axis='time', sr=sr3)
plt.colorbar()
plt.tight_layout()

S_filter_3 = librosa.decompose.nn_filter(S_full_3,aggregate=np.median, metric='cosine',width=int(librosa.time_to_frames(2, sr=sr3)))
S_filter_3 = np.minimum(S_full_3, S_filter_3)
# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
# Note: the margins need not be equal for foreground and background separation
margin_i_3, margin_v_3 = 2, 10
power_3 = 2
mask_i_3 = librosa.util.softmask(S_filter_3,margin_i_3 * (S_full_3 - S_filter_3),power=power_3)
mask_v_3 = librosa.util.softmask(S_full_3 - S_filter_3, margin_v_3 * S_filter_3,power=power_3)
# Once we have the masks, simply multiply them with the input spectrum
# to separate the components
S_foreground_3 = mask_v_3 * S_full_3
S_background_3 = mask_i_3 * S_full_3 
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full_3[:, idx_3], ref=np.max), y_axis='log', sr=sr3)
plt.title('Full spectrum : vandematram_shreya')
plt.colorbar()
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background_3[:, idx_3], ref=np.max),y_axis='log', sr=sr3)
plt.title('Background : vandematram_shreya')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground_3[:, idx_3], ref=np.max),y_axis='log', x_axis='time', sr=sr3)
plt.title('Foreground : vandematram_shreya')
plt.colorbar()
plt.tight_layout()
plt.show()




''' COMPARISON '''
voc_anuja= S_foreground_1
voc_lata= S_foreground_2
voc_shreya= S_foreground_3
print("Dimensions of vandematram_anuja song vocals:",voc_anuja.shape)
print("Dimensions of vandematram_lata song vocals:",voc_lata.shape)
print("Dimensions of vandematram_shreya song vocals:",voc_shreya.shape)
type(voc_anuja)

voc1=voc_anuja.flatten()
voc2=voc_lata.flatten()
voc3=voc_shreya.flatten()

vocl1=list(voc1)
vocl2=list(voc2)
vocl3=list(voc3)

dim1=len(vocl1)
dim2=len(vocl2)
dim3=len(vocl3)

large=max(dim1,dim2,dim3)
print(large)

diff1=large-dim1
diff2=large-dim2
diff3=large-dim3

for i in range (diff1):
    vocl1.append(0)

for j in range (diff2):
    vocl2.append(0)   
    
for k in range (diff3):
    vocl3.append(0)
    
len(vocl1)
len(vocl2)
len(vocl3)

from scipy.spatial.distance import euclidean
import time

start1=time.time()
diste12=distance.euclidean(vocl1, vocl2)
stop1=time.time()
print("Euclidean distance between vandemataram by anuja, lata:",diste12)

diste23=distance.euclidean(vocl2, vocl3)
print("Euclidean distance between vandemataram by shreya, lata:",diste23)
diste31=distance.euclidean(vocl1, vocl3)
print("Euclidean distance between vandemataram by anuja, shreya:",diste31)

#Similarity between vocals
s1=set(voc1)
s2=set(voc2)
s3=set(voc3)

m12=(len(s1&s2))
m23=(len(s3&s2))
m13=(len(s1&s3))
print("Similarity index between vandemataram by anuja, lata:",m12)
print("Similarity index between vandemataram by sherya, lata:",m23)
print("Similarity index between vandemataram by anuja, shreya:",m13)


from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

#dtw distance
start2=time.time()
distance0, path0 = fastdtw(vocl1, vocl2, dist=euclidean)
print(distance0)
stop2=time.time()

distance1, path1 = fastdtw(vocl2, vocl3, dist=euclidean)
print(distance1)

distance2, path2 = fastdtw(vocl1, vocl3, dist=euclidean)
print(distance2)


print("Dtw distance between vandemataram by anuja, lata:",distance0)
print("Dtw distance between vandemataram by shreya, lata:",distance1)
print("Dtw distance between vandemataram by anuja, shreya:",distance2)

#without padding using fastdw
v1=list(voc1)
v2=list(voc2)
v3=list(voc3)

start3=time.time()
dis12, path12 = fastdtw(v1, v2, dist=euclidean)
stop3=time.time()
print("Dtw distance without padding between vandemataram by anuja, lata:",dis12)

dis23, path23 = fastdtw(v3, v2, dist=euclidean)
print("Dtw distance without padding between vandemataram by anuja, lata:",dis23)
dis13, path13 = fastdtw(v1, v3, dist=euclidean)
print("Dtw distance without padding between vandemataram by anuja, lata:",dis13)

### Hamming method
from scipy.spatial.distance import hamming

start4=time.time()
disth1=distance.hamming(vocl1, vocl2)
stop4=time.time()

disth2=distance.hamming(vocl2, vocl3)
disth3=distance.hamming(vocl1, vocl3)

print("Hamming distance between vandemataram by anuja, lata:",disth1)
print("Hamming distance between vandemataram by shreya, lata:",disth2)
print("Hamming distance between vandemataram by anuja, shreya:",disth3)
print("time taken for hamming distance: ",(stop4-start4)/60," minutes")

print("=======================================================")
#comparison between methods
print("Time comparisons between different methods for case 1 ")
print("time taken for euclidean distance: ",(stop1-start1)/60," minutes")
print("time taken for dtw distance: ",(stop2-start2)/60," minutes")
print("time taken for dtw distance without padding: ",(stop3-start3)/60," minutes")
print("time taken for hamming distance: ",(stop4-start4)/60," minutes")
print("======================================")

#distance comparison between different methods
print("Comparison between songs by anuja & lata: ")
print("Euclidean distance= ", diste12)
print("DTW distance=", distance0 )
print("DTW distance without padding=", dis12 )
print("Hamming distance= ", disth1)
print("======================================")

print("Comparison between songs by shreya & lata: ")
print("Euclidean distance= ", diste23)
print("DTW distance=", distance1)
print("DTW distance without padding=", dis23)
print("Hamming distance= ", disth2)
print("======================================")

print("Comparison between songs by anuja & shreya: ")
print("Euclidean distance= ", diste31)
print("DTW distance=", distance2)
print("DTW distance without padding=", dis13)
print("Hamming distance= ", disth3)
print("======================================")

#Plotting seperate graphs
import wave
import numpy as np
import matplotlib.pyplot as plt

signal_wave_1 = wave.open('vandematramanuja.wav', 'r')
sample_rate = 16000
sig_1 = np.frombuffer(signal_wave_1.readframes(sample_rate), dtype=np.int16)
sig_1 = sig_1[:]
plt.figure(1)
plot_a = plt.subplot(211)
plot_a.plot(sig_1,color="r")
plot_a.set_xlabel('sample rate * time')
plot_a.set_ylabel('energy')

signal_wave_2 = wave.open('vandematramlata.wav', 'r')
sample_rate = 16000
sig_2 = np.frombuffer(signal_wave_2.readframes(sample_rate), dtype=np.int16)
sig_2 = sig_2[:]
plt.figure(2)
plot_b = plt.subplot(211)
plot_b.plot(sig_2,color="g")
plot_b.set_xlabel('sample rate * time')
plot_b.set_ylabel('energy')

signal_wave_3 = wave.open('vandematramshreya.wav', 'r')
sample_rate = 16000
sig_3 = np.frombuffer(signal_wave_3.readframes(sample_rate), dtype=np.int16)
sig_3 = sig_3[:]
plt.figure(3)
plot_c = plt.subplot(211)
plot_c.plot(sig_3,color="b")
plot_c.set_xlabel('sample rate * time')
plot_c.set_ylabel('energy')

#Combined Plots
plt.plot(sig_2,color="g")
plt.plot(sig_1,color="r")

plt.plot(sig_2,color="g")
plt.plot(sig_3,color="b")

############################################################################################
voc_anuja.shape
voc_lata.shape
import scipy
from scipy import spatial
from scipy.spatial import distance
from scipy.spatial.distance import cdist

A = np.reshape(voc_anuja, (-1, 2))
A.shape
voc_anuja.shape
voc_lata.shape
B = np.reshape(voc_lata, (-1, 2))
B.shape
C = np.reshape(voc_shreya, (-1, 2))
C.shape

distry=cdist(A, C, metric='hamming')

a=list(voc1)
b=list(voc2)
c=list(voc3)
ab=list(set(b) - set(a))
ab

###################################################################
voc1.shape
voc2.shape
one=voc1[:,np.newaxis]
one.shape
two=voc2[:,np.newaxis]

vone= one.transpose()
vone.shape

vtwo= two.transpose()
vtwo.shape

diste12=distance.euclidean(vone, vtwo)
np.dot(vone,vtwo)
distry=cdist(one, two, metric='euclidean')

##############################################
from scipy.spatial.distance import hamming
#disth1=cdist(vocl1, vocl2, metric='hamming')
start4=time.time()
disth1=distance.hamming(vocl1, vocl2)
stop4=time.time()
print("Hamming distance =" ,disth1)
print((stop4-start4)/60)

disth1=distance.hamming(one, two)
len(vone)
len(vtwo)
######################################################################

flag=0
array1=[1,2,3,4]
array2=[5,2,1,4,5,6,7,1]
for i in range(max(len(array1),len(array2))):
    if array1[i]==array2[i]:
        flag+=1
    else:
        continue
print(flag)
