# -*- coding: utf-8 -*-
"""
Created on Fri May 15 01:09:04 2020

@author: prach
"""

import os
import shutil
from flask import Flask
from spleeter.commands.separate import entrypoint
from spleeter.utils.configuration import load_configuration
from spleeter.audio.adapter import get_audio_adapter
from spleeter.separator import Separator
from spleeter.utils.estimator import create_estimator, to_predictor

def get_parameters():
    """
    TODO: load config from json file for following or take from user
    separate -i spleeter/audio_example.mp3 -p spleeter:2stems -o output
    """
    parameters = {"MWF":False, 
                  "audio_adapter":None, "bitrate":'128k',
                "codec":'wav', "command":'separate', "configuration":'spleeter:2stems',
                "duration":300.0, "filename_format":'{filename}/{instrument}/{part}.{codec}',
                "inputs":['audio_example.mp3'], "offset":0.0, "output_path":'output',
                "verbose":False}
    return parameters





if __name__ == "__main__":

    arguments = get_parameters()
    parameters = load_configuration(arguments["configuration"])
    load_audio_adapter(arguments)
    load_estimator(arguments, parameters)

    #process_file("audio-example-2.mp3")
