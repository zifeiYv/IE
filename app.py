# -*- coding: utf-8 -*-
import warnings
import logging
from collections import defaultdict

from flask import Flask, request, jsonify
from ner.run_predict import predict as ner_predict
from ere.run_predict import predict as ere_predict
from utils import check_path, extract_sentences, beautify_results
from log import handler

warnings.filterwarnings('ignore')
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


@app.route('/ie/ner', methods=['POST'])
def ner():
    sentence = request.json['sentence']
    pred = ner_predict(sentence)
    return jsonify({"results": pred})


@app.route('/ie/ere', methods=['POST'])
def ere():
    sentence = request.json['sentence']
    pred = ere_predict(sentence)
    return jsonify({'results': pred})


@app.route('/ie/ie', methods=['POST'])
def ie():
    sentence = request.json['sentence']
    # todo
    return sentence


@app.route('/ie/ner_from_path', methods=['POST'])
def ner_from_path():
    """
    给定一个文件的绝对路径，对其中的内容进行NER；
    或者给定一个目录，算法会遍历其中的可解析文件，然后逐个提取内容并进行NER。

    注意，算法不支持嵌套文件夹的遍历，即如果所指定路径含子文件夹，则不会读取子文件夹的内容。
    """
    paras = request.json
    res = from_path(paras)
    if res.get('state') == 0:
        return jsonify(res)
    else:
        file_sentences = res
    res = defaultdict(list)
    logger.info('开始抽取实体...')
    counter = 1
    for ft in file_sentences:
        logger.info(f'  {counter}/{len(file_sentences)}：当前处理`{ft}`格式文件')
        inner_counter = 1
        for file_name in file_sentences[ft]:
            sentences = file_sentences[ft][file_name]
            logger.info(f'    {inner_counter}/{len(file_sentences[ft])}: `{file_name}`共{len(sentences)}个句子')
            res[file_name] = beautify_results(ner_predict(sentences))
            inner_counter += 1
        counter += 1
    logger.info('Done')
    return jsonify({'results': dict(res)})


@app.route('/ie/ere_from_path', methods=['POST'])
def ere_from_path():
    """
    给定一个文件的绝对路径，对其中的内容进行ERE；
    或者给定一个目录，算法会遍历其中的可解析文件，然后逐个提取内容并进行ERE。

    注意，算法不支持嵌套文件夹的遍历，即如果所指定路径含子文件夹，则不会读取子文件夹的内容。
    """
    paras = request.json
    res = from_path(paras)
    if res.get('state') == 0:
        return jsonify(res)
    else:
        file_sentences = res
    res = defaultdict(list)
    logger.info('开始抽取关系...')
    counter = 1
    for ft in file_sentences:
        logger.info(f'  {counter}/{len(file_sentences)}：当前处理`{ft}`格式文件')
        inner_counter = 1
        for file_name in file_sentences[ft]:
            sentences = file_sentences[ft][file_name]
            logger.info(f'    {inner_counter}/{len(file_sentences[ft])}: `{file_name}`共{len(sentences)}个句子')
            res[file_name] = ere_predict(sentences)
            inner_counter += 1
        counter += 1
    logger.info('Done')
    return jsonify({'results': dict(res)})


def from_path(paras: dict):
    """根据参数进行预处理"""
    path = paras['file_or_directory_path']
    file_type = paras['file_type']
    logger.info(f'获取参数：{paras}')
    file_dict = check_path(path, file_type)
    if len(file_dict) == 0:
        err_msg = f'{path}不是一个文件/不是一个路径/路径下不包含可解析的文件！'
        logger.error(err_msg)
        return {'state': 0, 'msg': err_msg}
    file_sentences = extract_sentences(file_dict)
    return file_sentences


if __name__ == '__main__':
    app.run()
