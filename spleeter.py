# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:27:07 2020

@author: prach
"""

import spleeter
from spleeter.separator import Separator
from ffmpeg import _run
from ffmpeg import probe
import FFMPEGProcessAudioAdapter
# import spleeter.Seperator
# from spleeter.separator import Seperator
split=spleeter.separator.Separator('spleeter:2stems')


audio_generator= spleeter.audio.adapter.get_default_audio_adapter()
sample_rate = 44100

waveform,_ = audio_generator.load('C:\\Users\\prach\\OneDrive\\Desktop\\Projects\\music audio seperationmyaudiofile', sample_rate=sample_rate)
prediction = Separator.separate(waveform)

Separator.separate_to_file()
Separator.separate_to_file('C:\\Users\\prach\\OneDrive\\Desktop\\Projects\\music audio\\seperation\\myaudiofile','C:\\Users\\prach\\OneDrive\\Desktop\\Projects\\music audio seperation\\')
 
help(Separator.separate())
help(Separator.separate_to_file)
Separator.separate_to_file("myaudiofile","C:\\Users\\prach\\OneDrive\\Desktop\\Projects\\music audio seperation")


spleeter.seperator.separate_to_file("mysudiofile","C:\\Users\\prach\\OneDrive\\Desktop\\Projects")

spleeter.seperationmyaudiofile

spleeter.seperator.Seperator.separate_to_file("mysudiofile","C:\\Users\\prach\\OneDrive\\Desktop\\Projects")

song="C:\\Users\\prach\\OneDrive\\Desktop\\myaudiofile"
split="C:\\Users\\prach\\OneDrive\\Desktop\\Projects"
Separator.separate_to_file(audio_generator.load,audio_descriptor=song,destination=split)
