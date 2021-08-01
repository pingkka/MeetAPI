import pickle
import re
from timeit import default_timer as timer
from datetime import timedelta

import librosa
import numpy as np
import torch
from transformers import AutoTokenizer

import hw_text_model


class LanoiceClassification():
    def __init__(self):
        self.labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]

        # 음성 모델 파일명
        self.filename = 'xgb_model3004.model'
        # 음성 모델 불러오기
        self.loaded_model = pickle.load(open(self.filename, 'rb'))

        # 텍스트 모델 초기값
        self.none_words = ["안싫", "안 싫", "안무서", "안놀람", "안놀랐", "안행복", "안기뻐", "안빡", "안우울", "안짜증", "안깜짝", "안무섭"]
        self.pass_words = ["안좋", "안 좋"]
        self.senti_loss = [5.0, 3.5, 4.0, 8.0, 6.0, 9.5]
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        # GPU 사용 여부
        self.device = torch.device("cpu")

        # 텍스트 모델 불러오기
        self.model = hw_text_model.HwangariSentimentModel.from_pretrained("Kyuyoung11/haremotions-v3").to(self.device)

    def classify(self, audio_path, text):

        ########################### TESTING ###########################
        print("audio classify speed")
        start = timer()
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
        # y_chunk_model1 = self.loaded_model.predict(x_chunk)
        y_chunk_model1_proba = self.loaded_model.predict_proba(x_chunk)
        index = np.argmax(y_chunk_model1_proba)
        end = timer()
        print(timedelta(seconds=end - start))

        # print("----------------------------")
        # print(f'Review text : {text}')
        # print("<Audio Accuracy>")
        # for proba in range(0, len(y_chunk_model1_proba[0])):
        #     print(self.labels[proba] + " : " + str(y_chunk_model1_proba[0][proba]))
        #
        # print('\nEmotion:', self.labels[int(index)])

        print("\ntext classify speed")
        start = timer()
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

        label_loss_str = str(output).split(",")

        label_loss = [float(x.strip().replace(']', '')) for x in label_loss_str[1:7]]
        # print("\n<Text Loss>")

        pre_result = int(re.findall("\d+", str(prediction))[0])

        # 손실함수 값이 senti_loss 값보다 높아야 해당 감정으로 분류
        result = 0
        if label_loss[pre_result - 1] >= self.senti_loss[pre_result - 1]:
            result = pre_result

        end = timer()
        print(timedelta(seconds=end - start))

        '''
        # 안이 들어간 말로 결과가 나왔을 경우 가장 큰 값을 무시함 or 아예 무감정으로 분류되도록 함
        for i in self.none_words:
            if i in text:
                result = 0
        for j in self.pass_words:
            if j in text:
                label_loss[pre_result - 1] = 0
                result = label_loss.index(max(label_loss)) + 1
        '''

        # for i in range(0, 6):
        #     print(self.labels[i + 1], ":", label_loss[i])
        # print(f'Sentiment : {self.labels[result]}')

        # 결과 합산 (값 기반 계산)
        if (index == 0 or (result == 0 and pre_result == 4) or (result == 0 and pre_result == 5) or (
                result == 0 and pre_result == 6)):
            total_result = -1
        elif (index == pre_result):
            total_result = index - 1

        else:
            text_score = []
            audio_score = []
            total_score = []
            label_loss[4] = label_loss[4] * 0.7
            label_loss[5] = label_loss[5] * 0.7
            for i in range(0, len(label_loss)):
                text_score.append(label_loss[i] / (sum(label_loss) + 10))

                audio_score.append(y_chunk_model1_proba[0][i + 1] - 0.25)

            for i in range(0, len(audio_score)):
                total_score.append(float(audio_score[i]) + float(text_score[i]))
            # print(total_score)

            total_result = total_score.index(max(total_score))

        '''
        # 결과 합산 (값 기반 계산)
        if (index == 0 or (result == 0 and pre_result == 5) or (result == 0 and pre_result == 6)):
            print("none")
            total_result = -1
        elif (index == pre_result):
            print("same")
            total_result = index -1

        else:
            print("score")
            text_score = []
            audio_score = []
            total_score = []
            for i in range(0, len(label_loss)):
                text_score.append(label_loss[i])
                audio_score.append(y_chunk_model1_proba[0][i + 1] * 10)

            for i in range(0, len(audio_score)):
                total_score.append(float(audio_score[i]) + float(text_score[i]))
            print(total_score)

            total_result = total_score.index(max(total_score))
        '''

        '''
        #순위 기반 점수 측정
        if (index == 0 or result == 0):
            total_result = -1
        else :
            # 음성 순위 기반 점수 계산
            audio_rank = [0, 0, 0, 0, 0, 0]
            new_proba = y_chunk_model1_proba[0]
            new_proba[0] = 0


            for i in range(6, 0, -1):
                rank = np.argmax(new_proba)
                audio_rank[rank - 1] += i
                new_proba[rank] = 0

            # 텍스트 순위 기반 점수 계산
            text_rank = [0, 0, 0, 0, 0, 0]
            new_text_loss = label_loss
            for i in range(6, 0, -1):
                rank = new_text_loss.index(max(new_text_loss))
                new_text_loss[rank] = -100

                text_rank[rank] += (i * 2)
            total_score=[]
            #print(audio_rank)
            #print(text_rank)
            for i in range(0, len(audio_rank)-1):
                total_score.append(audio_rank[i] + text_rank[i])


            print(total_score)
            total_result = total_score.index(max(total_score))
        '''

        print("Result : " + self.labels[total_result + 1])
        print("---------------------------------")

        return self.labels[total_result + 1]
