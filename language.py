from flask import request
from flask_restx import Resource, Namespace

import text_model

Language = Namespace('Language')
string = text_model.textClassification()


@Language.route('/get-emotion-from-text')
class LanguagePost(Resource):
    def post(self):
        try:
            text = request.files['file'].read().decode('utf-8')
            emotion = string.textClassification(text)
            print(text)
            return {'emotion': emotion}

        except Exception as e:
            return {'error': str(e)}
