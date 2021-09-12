import re

import torch
from transformers import AutoTokenizer

import hw_text_model


class textClassification():
    def __init__(self):
        self.labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
        self.none_words = ["안싫", "안 싫", "안무서", "안놀람", "안놀랐", "안행복", "안기뻐", "안빡", "안우울", "안짜증", "안깜짝", "안무섭"]
        self.pass_words = ["안좋", "안 좋"]
        self.senti_loss = [5.0, 4.0, 6.5, 6.5, 9.0, 8.0]
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        # GPU 사용
        self.device = torch.device("cpu")
        self.model = hw_text_model.HwangariSentimentModel.from_pretrained("Kyuyoung11/haremotions-v2").to(self.device)

    def textClassification(self, text):

        # enc = tokenizer.encode_plus(text)
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

        # print(f'Review text : {text}')

        pre_result = int(re.findall("\d+", str(prediction))[0])
        # 손실함수 값이 4.0이상인게 없으면 무감정(none)으로 분류
        result = 0
        if label_loss[pre_result - 1] >= self.senti_loss[pre_result - 1]:
            result = pre_result

        # 안이 들어간 말로 결과가 나왔을 경우 가장 큰 값을 무시함 or 아예 무감정으로 분류되도록 함
        for i in self.none_words:
            if i in text:
                result = 0
        for j in self.pass_words:
            if j in text:
                label_loss[pre_result - 1] = 0
                result = label_loss.index(max(label_loss)) + 1

        # print(f'Sentiment : {labels[result]}')

        # print("\n<감정 별 손실 함수 값>")
        # for i in range(0,6):

        # print(labels[i+1], ":", label_loss[i])

        return str(self.labels[result])
