# -*- coding: utf-8 -*-
# Author: sunjw
import re


class BaseExtract:
    separators = ['。', '；', '？', '！']
    p = re.compile(str(separators))

    def get_list(self, file_path):
        raise NotImplementedError

    @classmethod
    def match_separators(cls, string):
        for i in cls.p.finditer(string):
            yield i.span()[1]
