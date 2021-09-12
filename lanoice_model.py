########################################################
#텍스트와 음성 모두 예측하는 코드
########################################################

import pickle
import re

import librosa
import numpy as np
import torch
from transformers import AutoTokenizer

import hw_text_model



class LanoiceClassification():
    def __init__(self):
        self.labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
        self.gender = ["male", "female"]

        #XGBoost 모델 사용시
        # 음성 모델 파일명
        #self.filename = 'audio_model/xgb_4_300_0.1_6.model'
        # 음성 모델 불러오기
        #self.audio_model = pickle.load(open(self.filename, 'rb'))

        # #Keras Sequential 모델 사용시
        # self.audio_model = load_model('audio_model/audio_4_unit25_0.1.h5')

        # gender 모델 파일명
        self.gender_file = 'xgb_1_300_0.1_6.model'

        # gender 모델 불러오기
        self.gender_model = pickle.load(open(self.gender_file, 'rb'))

        # 텍스트 모델 초기값

        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        # GPU 사용 여부
        self.device = torch.device("cuda")

        # 텍스트 모델 불러오기
        self.model = hw_text_model.HwangariSentimentModel.from_pretrained("Kyuyoung11/haremotions-v5").to(self.device)

    def classify(self, audio_path, text):

        ########################### TESTING ###########################
        # test_file_path = "5_wav/5f05fb0bb140144dfcff0184.wav"
        X, sr = librosa.load(audio_path, sr=None)
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
        y_chunk_gender_proba = self.gender_model.predict_proba(x_chunk)
        gender_index = np.argmax(y_chunk_gender_proba)
        if (gender_index == 0):
            filename = 'xgb_5_300_0.1_6_m.model'
            # 음성 모델 불러오기
            audio_model =  pickle.load(open(filename, 'rb'))
        elif (gender_index == 1):
            filename = 'xgb_1_300_0.1_6_f.model'
            # 음성 모델 불러오기
            audio_model = pickle.load(open(filename, 'rb'))


        y_chunk_model1_proba = audio_model.predict_proba(x_chunk)
        #y_chunk_model1_proba = audio_model.predict(x_chunk)
        print(y_chunk_model1_proba)
        index = np.argmax(y_chunk_model1_proba)


        print("----------------------------")
        print(f'Review text : {text}')
        print("<Audio Accuracy>")
        for proba in range(0, len(y_chunk_model1_proba[0])):
            print(self.labels[proba] + " : " + str(y_chunk_model1_proba[0][proba]))

        print('\nEmotion:', self.labels[int(index)])


        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        # print(tokenizer.tokenize(tokenizer.decode(enc["input_ids"])))

        self.model.eval()

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        output = self.model(input_ids.to(self.device), attention_mask.to(self.device))[0]
        _, prediction = torch.max(output, 1)
        #print(output)

        label_loss_str = str(output).split(",")
        label_loss_str[0] = label_loss_str[0][9:]
        label_loss = [float(x.strip().replace(']', '')) for x in label_loss_str[0:7]]
        print("\n<Text Loss>")


        #pre_result = int(re.findall("\d+", str(prediction))[0])
        result = int(re.findall("\d+", str(prediction))[0])





        for i in range(0, len(label_loss)):
            print(self.labels[i], ":", label_loss[i])
        print(f'Sentiment : {self.labels[result]}')

        text_score = []
        audio_score = []
        total_score = []


        #결과 합산(1) - 단순 합산
        for i in range(0, len(label_loss)):
            text_score.append(label_loss[i] / (sum(label_loss) + 10))
            audio_score.append(y_chunk_model1_proba[0][i] - 0.2)
        for i in range(0, len(audio_score)):
            total_score.append(float(audio_score[i]) + float(text_score[i]))
        #print(text_score, audio_score)
        print(total_score)
        if (total_score[0] >= 0.7):
            total_result = total_score.index(max(total_score))
        else:
            total_result = total_score.index(max(total_score[1:]))





        # #결과 합산(2) - 순위 합산
        # text_score = label_loss
        # audio_proba = y_chunk_model1_proba[0].tolist()
        # audio_score = audio_proba
        #
        # text_rank = []
        # audio_rank = []
        # for i in range(5):
        #     text_max = label_loss.index(max(text_score))
        #     audio_max = audio_proba.index(max(audio_score))
        #     text_rank.append(text_max)
        #     audio_rank.append(audio_max)
        #
        #     text_score[text_max] = -100
        #     audio_score[audio_max] = -100
        # print(text_rank, audio_rank)
        #
        # result_score = [0] * len(label_loss)
        # for i in range(len(text_rank)):
        #     result_score[text_rank[i]] += (len(text_rank)-i)*2
        #     result_score[audio_rank[i]] += (len(text_rank)-i)*2
        # print(result_score)
        #
        # total_result = result_score.index(max(result_score))







        # print("Result : " + self.labels[total_result])
        print("---------------------------------")

        # return self.labels[total_result], self.gender[gender_index]
        return self.labels[total_result]