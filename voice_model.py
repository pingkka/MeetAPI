import numpy as np
import librosa
import pickle

import wave


class audioClassification():
    def __init__(self):
        self.emotions = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]

        # 파일명
        self.filename = 'xgb_model.model'

        # 모델 불러오기
        self.loaded_model = pickle.load(open(self.filename, 'rb'))

    def classify(self, file):
        ########################### PREDICTION ###########################
        X, sr = librosa.load(file, sr=None)
        stft = np.abs(librosa.stft(X))

        ############# EXTRACTING AUDIO FEATURES #############
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

        mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)

        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        x_chunk = np.array(features)
        x_chunk = x_chunk.reshape(1, np.shape(x_chunk)[0])
        y_chunk_model1 = self.loaded_model.predict(x_chunk)
        y_chunk_model1_proba = self.loaded_model.predict_proba(x_chunk)
        index = np.argmax(y_chunk_model1)

        # print("-----<Accuracy>------")
        # for proba in range(0, len(y_chunk_model1_proba[0])):
        #    print(self.emotions[proba]+  " : " + str(y_chunk_model1_proba[0][proba]))

        # print('\nEmotion:',self.emotions[int(y_chunk_model1[0])])
        return str(self.emotions[int(y_chunk_model1[0])])
