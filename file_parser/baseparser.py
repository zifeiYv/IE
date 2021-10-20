# -*- coding: utf-8 -*-
# Author: sunjw


class BaseExtract:
    separators = ['。', '；', '？', '！']

    def get_list(self, file_path):
        raise NotImplementedError
