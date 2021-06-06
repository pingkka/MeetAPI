# -*- coding:utf-8 -*-
import os
import wave

from flask import request
from flask_restx import Resource, Namespace
from werkzeug.utils import secure_filename

import lanoice_model
import usage_info
import datetime
import pytz

global usage_info_list

Lanoice = Namespace('Lanoice')

lanoice = lanoice_model.LanoiceClassification()

usage_info_list = list()


@Lanoice.route('/get-emotion')
class LanguageVoicePost(Resource):
    def post(self):
        info = usage_info.UsageInfo()
        now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
        info.ip = request.remote_addr
        info.time = now
        try:

            text = request.files['text_file'].read().decode('utf-8')
            print(text)

            pcm_file = request.files['audio_file']
            pcm_file.save(secure_filename(pcm_file.filename))
            split = pcm_file.filename.split('.pcm')
            wav_filename = split[0] + '.wav'
            pcm2wav(pcm_file.filename, wav_filename)

            emotion = lanoice.classify(wav_filename, text)

            info.text = text
            info.emotion = emotion

            usage_info_list.append(info)

            remove_file(pcm_file.filename)
            remove_file(wav_filename)

            return {'emotion': emotion}

        except Exception as e:
            info.text = str(e)
            info.emotion = "error"

            usage_info_list.append(info)
            return {'error': str(e)}


def pcm2wav(pcm_file, wav_file, channels=1, bit_depth=16, sampling_rate=16000):
    # Check if the options are valid.
    if bit_depth % 8 != 0:
        raise ValueError("bit_depth " + str(bit_depth) + " must be a multiple of 8.")

    # Read the .pcm file as a binary file and store the data to pcm_data
    with open(pcm_file, 'rb') as opened_pcm_file:
        pcm_data = opened_pcm_file.read();

        obj2write = wave.open(wav_file, 'wb')
        obj2write.setnchannels(channels)
        obj2write.setsampwidth(bit_depth // 8)
        obj2write.setframerate(sampling_rate)
        obj2write.writeframes(pcm_data)
        obj2write.close()


def remove_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)
