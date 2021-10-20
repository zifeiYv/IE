# -*- coding: utf-8 -*-
# Author: sunjw
import os
import logging
from logging.handlers import RotatingFileHandler


def get_project_pys():
    """获取项目的所有py文件"""
    py_dir = os.path.split(os.path.realpath(__file__))[0]
    pys = []  # 记录项目所有的py文件
    for _, _, i in os.walk(py_dir):
        pys.extend([j for j in i if j.endswith('py')])
    return pys


def filter_fn(record):
    if record.filename in pys:
        return True
    return False


pys = get_project_pys()

# 设置日志输出的样式
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(lineno)d - %(message)s')
# 设置日志在控制台输出
handler = RotatingFileHandler(filename='app.log', maxBytes=5 * 1024 * 1024, backupCount=10, encoding='utf-8')
# 给该handler绑定formatter
handler.setFormatter(formatter)
# 给该handerl绑定自定义的过滤器
handler.addFilter(filter_fn)

