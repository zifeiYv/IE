# -*- coding: utf-8 -*-
# Author: sunjw
from .baseparser import BaseExtract


class TextExtract(BaseExtract):

    @classmethod
    def get_list(cls, file_path):
        string = ""
        with open(file_path) as f:
            for line in f.readlines():
                string += line.strip()
        start = 0
        sentences = []
        for i in cls.match_separators(string):
            sentences.append(string[start: i])
            start = i
        return sentences
