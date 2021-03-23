# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:54:35 2020

@author: prach
"""
import pydub
from pydub import AudioSegment
from pydub.playback import play
import librosa

# read in audio file and get the two mono tracks
audio='myaudiofile.mp3'
sound_stereo = pydub.AudioSegment.from_file(audio, format="mp3")
sound_monoL = sound_stereo.split_to_mono()[0]
sound_monoR = sound_stereo.split_to_mono()[1]

# Invert phase of the Right audio file
sound_monoR_inv = sound_monoR.invert_phase()

# Merge two L and R_inv files, this cancels out the centers
sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)

# Export merged audio file
fh = sound_CentersOut.export(myaudioFile_CentersOut, format="mp3")
fh = sound_CentersOut.export(sound_monoR_inv, format="mp3")

