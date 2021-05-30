import os
import wave

from flask import request
from flask_restx import Resource, Namespace
from werkzeug.utils import secure_filename

import voice_model

Voice = Namespace('Voice')
audio = voice_model.audioClassification()


@Voice.route('/get-emotion-from-voice')
class VoicePost(Resource):
    def post(self):
        try:
            pcm_file = request.files['file']
            pcm_file.save(secure_filename(pcm_file.filename))
            split = pcm_file.filename.split('.pcm')
            wav_filename = split[0] + '.wav'
            pcm2wav(pcm_file.filename, wav_filename)
            # print(wav_file)
            emotion = audio.classify(wav_filename)

            remove_file(pcm_file.filename)
            remove_file(wav_filename)

            return {'emotion': emotion}

        except Exception as e:
            return {'error': str(e)}

        # pcm 파일을 wav로 바꿈
        # The parameters are prerequisite information. More specifically,
        # channels, bit_depth, sampling_rate must be known to use this function.


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
