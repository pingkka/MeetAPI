from flask import request
from flask_restx import Resource, Api, Namespace, reqparse

Language = Namespace('Language')


@Language.route('/get-emotion')
class LanguagePost(Resource):
    def post(self):
        try:
            file = request.files['file'].read().decode('utf-8')
            print(type(file))
            print(file)

        except Exception as e:
            return {'error': str(e)}
