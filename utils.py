# -*- coding: utf-8 -*-
# Utility functions
# Author: sunjw
from __future__ import annotations

import os
import logging

from collections import defaultdict
from file_parser import PdfExtract, TextExtract
from config import NER_LABELS, ERE_LABELS

# 支持的文件类型
# 对应于每种类型的文件，均需要一个特定的解析器
parsers = {
    'pdf': PdfExtract,
    'txt': TextExtract
}

logger = logging.getLogger(__name__)


def check_path(path: str, file_type: str = None):
    """检查传入的文件是否有效"""
    file_dict = defaultdict(list)
    if not os.path.isabs(path):
        logger.error(f'{path} must be an absolute path.')
        raise Exception(f'{path} must be an absolute path.')
    if os.path.isfile(path):
        logger.info('A file found!')
        ft = os.path.split(path)[1].split('.')[1]
        if ft not in parsers:
            logger.error(f'Unsupported file type: {ft}, you\'re allowed to pass files in'
                         f'{list(parsers.keys())} types.')
            raise Exception(f'Unsupported file type: {ft}, you\'re allowed to pass files in'
                            f'{list(parsers.keys())} types.')
        return {ft: [path]}
    elif os.path.isdir(path):
        logger.info('A directory found!')
        file_type = None if file_type == "" else file_type
        if file_type is not None:
            if file_type not in parsers:
                logger.warning(f'The passing file type: `{file_type}` is not in '
                               f'supported file type list: `{list(parsers.keys())}` '
                               f'and will be ignored.')
                file_type = None
        for i in os.listdir(path):
            for ft in parsers:
                if file_type is not None and ft != file_type:
                    continue
                if (not i.startswith('.')) and (i.endswith(ft)):
                    file_dict[ft[1:]].append(os.path.join(path, i))
        return file_dict
    else:
        raise FileNotFoundError(f"{path} is neither a file path or a directory path.")


def extract_sentences(file_dict: dict):
    """给出传入的文件，抽取其中的句子"""
    logger.info(f'Extracting sentence from the given {file_dict}...')
    file_sentences = defaultdict(dict)
    counter = 1
    for ft in file_dict:
        logger.debug(f'  {counter}/{len(file_dict)}...')
        extractor = parsers[ft]
        inner_counter = 1
        for i in file_dict[ft]:
            logger.debug(f'    {inner_counter}/{len(file_dict[ft])}')
            file_sentences[ft][i] = extractor.get_list(i)
            inner_counter += 1
        counter += 1
    logger.info(f'Done')
    return file_sentences


def beautify_results(res: str | list, type_: str = 'ner'):
    """对模型输出的结果进行美化"""
    if isinstance(res, str):
        res = [res]
    if type_ == 'ner':
        result = defaultdict(set)
        for r in res:
            line_res = _beautify_ner(r, NER_LABELS)
            for key in line_res:
                result[key] = result[key].union(line_res[key])
        final = {}
        for key in result:
            final[key] = list(result[key])
        return final
    elif type_ == 'ere':
        raise NotImplementedError
    else:
        raise NotImplementedError


def _beautify_ner(res: str, label: list | dict):
    """Written by wangsha"""
    result = res.split(')(')
    analysis_dict = defaultdict(set)
    for i in range(len(result)):
        string = result[i].replace('(', '') \
            .replace("'", '') \
            .replace(" ", '') \
            .replace(")", '')
        result_split = string.split(',')
        if result_split[1] in label:
            analysis_dict[result_split[1]].add(result_split[0])
    return analysis_dict
