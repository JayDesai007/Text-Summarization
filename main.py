from flask import Flask,request,Response
from lsa import summarize
from nltk_ import summarize_
from textrank import extract_sentences
from flask_cors import CORS


app = Flask(__name__,static_folder='static',static_url_path='/')
CORS(app)


@app.route('/')
def hello_world():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def serve_static(path):
    return app.send_static_file(path)

@app.route('/api/lsa',methods=['POST'])
def lsa():
    body = request.get_json()
    return summarize(body['data'],k=int(body['expectedLen']))

@app.route('/api/nltk',methods=['POST'])
def nltk_():
    body = request.get_json()
    return summarize_(body['data'],k=int(body['expectedLen']))

@app.route('/api/textrank',methods=['POST'])
def textrank():
    body = request.get_json()
    return extract_sentences(body['data'],k=int(body['expectedLen']))

if __name__ == '__main__':
    app.run()
