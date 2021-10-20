# -*- coding: utf-8 -*-
# Author: sunjw
import pdfplumber
from .baseparser import BaseExtract


class PdfExtract(BaseExtract):

    @classmethod
    def get_list(cls, file_path):
        """Written by wangsha"""
        pdf_str = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                if page.extract_text() is not None:
                    pdf_str += page.extract_text()
            pdf_str = pdf_str.replace('\n', '')
            pdf_str = pdf_str.replace(',', '')
            for i in cls.separators:
                pdf_str = pdf_str.replace(i, i + '\n')
            pdf_list = pdf_str.split('\n')           # split分割返回值列表
            return pdf_list
