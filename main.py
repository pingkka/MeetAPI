from flask import Flask
from flask_restx import Api, Resource

from language import Language

app = Flask(__name__)  # Flask 애플리케이션 생성 코드
api = Api(app)  # Flask 객체에 Api 객체 등록

api.add_namespace(Language, '/api')


@api.route('/help')
class HelloAPI(Resource):
    def get(self):
        message = {'msg': "Hwangari's api for MEET Application",
                   'help': "/api/get-emotion"}
        return message


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
