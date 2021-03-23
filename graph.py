# -*- coding: utf-8 -*-
"""
Created on Fri May 15 02:18:43 2020

@author: prach
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.output
import librosa.display

#load audio
y, sr = librosa.load('myaudiofile.mp3', duration=195)

# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

#plot slice
idx = slice(*librosa.time_to_frames([10, 15], sr=sr))
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
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground')
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

import numpy as np
import sounddevice as sd
import time
sps = 44100
freq_hz = 440.0
duration_s = 5.0
atten = 0.3
# NumpPy magic to calculate the waveform
each_sample_number = S_foreground.arange(duration_s * sps)
waveform = np.sin(2 * np.pi * each_sample_number * freq_hz / sps)
waveform_quiet = waveform * atten

# Play the waveform out the speakers
sd.play(waveform_quiet, sps)
time.sleep(duration_s)
sd.stop()