from flask import request
from flask_restx import Resource, Api, Namespace, reqparse
import emotion_model

Language = Namespace('Language')


@Language.route('/get-emotion')
class LanguagePost(Resource):
    def post(self):
        try:
            text = request.files['file'].read().decode('utf-8')
            print(text)
            emotion = emotion_model.textClassification()
            return {'emotion': emotion.textClassification(text)}

        except Exception as e:
            return {'error': str(e)}
