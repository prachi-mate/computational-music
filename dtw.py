# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:02:33 2020

@author: HELLO
"""

from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# Read stored audio files for comparison
fs1, data1 = wavfile.read("C:/Users/hp/Downloads/doors-and-corners-kid_thats-where-they-get-you.wav")
fs2, data2 = wavfile.read("C:/Users/hp/Downloads/doors-and-corners-kid_thats-where-they-get-you-2.wav")
fs3, data3 = wavfile.read("C:/Users/hp/Downloads/you-walk-into-a-room-too-fast_the-room-eats-you.wav")
fs4, data4 = wavfile.read("C:/Users/hp/Downloads/doors-and-corners-kid.wav")

# Take the max values along axis
data1 = np.amax(data1, axis=1)
data2 = np.amax(data2, axis=1)
data3 = np.amax(data3, axis=1)
data4 = np.amax(data4, axis=1)

#from scipy.io import wavfile
#from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

# Read stored audio files for comparison
#fs, data = wavfile.read("/dbfs/folder/clip1.wav")

# Set plot style
plt.style.use('seaborn-whitegrid')

# Create subplots
ax = plt.subplot(2, 2, 1)
ax.plot(data1, color='#67A0DA')
...

# Display created figure
fig=plt.show()
display(fig)


#from fastdtw import fastdtw

# Distance between clip 1 and clip 2
#distance = fastdtw(data_clip1, data_clip2)[0]
distance = fastdtw(data1, data2)[0]
print("The distance between the two clips is %s" % distance)
