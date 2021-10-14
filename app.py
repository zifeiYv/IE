# -*- coding: utf-8 -*-
import warnings

from flask import Flask, request, jsonify
from ner.run_predict import predict as ner_predict
from ere.run_predict import predict as nre_predict
warnings.filterwarnings('ignore')
app = Flask(__name__)


@app.route('/ie/ner', methods=['POST'])
def ner():
    sentence = request.json['sentence']
    pred = ner_predict(sentence)
    return jsonify({"results": pred})


@app.route('/ie/ere', methods=['POST'])
def ere():
    sentence = request.json['sentence']
    pred = nre_predict(sentence)
    return jsonify({'results': pred})


@app.route('/ie/ie', methods=['POST'])
def ie():
    sentence = request.json['sentence']
    # todo
    return sentence


if __name__ == '__main__':
    app.run()
