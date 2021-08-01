# -*- coding:utf-8 -*-
import json

from flask import Flask, make_response
from flask_restx import Api, Resource

import lanoice
from language import Language
from lanoice import Lanoice
from voice import Voice

app = Flask(__name__)  # Flask 애플리케이션 생성 코드
app.config['JSON_AS_ASCII'] = False
api = Api(app)  # Flask 객체에 Api 객체 등록

api.add_namespace(Language, '/api')
api.add_namespace(Voice, '/api')
api.add_namespace(Lanoice, '/api')


@api.route('/help')
class HelloAPI(Resource):
    def get(self):
        message = "<H1>Hwangari's api for MEET Application</H1>" \
                  + "<br/>" \
                  + "<H3>INFO URL<H3/>" \
                  + "<br/>" \
                  + "<H4>Get Emotion from Text and Voice<br/>/api/get-emotion</H4>" \
                  + "<br/>" \
                  + "<H4>Get Emotion from Text<br/>/api/get-emotion-from-text</H4>" \
                  + "<br/>" \
                  + "<H4>Get Emotion from Voice<br/>/api/get-emotion-from-voice</H4>" \
                  + "<br/>" \
                  + "<H5>If you have any questions, please contact lsyaran99@hansung.ac.kr</H5>"
        res = make_response(message)
        return res


@api.route('/admin')
class AdminPage(Resource):
    def get(self):
        message = ""
        for info in lanoice.usage_info_list:
            msg = {'ip': info.ip,
                   'text': info.text,
                   'emotion': info.emotion,
                   'time': str(info.time)}
            message = message + json.dumps(msg, sort_keys=False, ensure_ascii=False, indent=4) + "<br/>"
            # print(msg)

        res = make_response(message)
        return res


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
