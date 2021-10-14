# -*- coding: utf-8 -*-
# Author: sunjw
import os

from paddlenlp.datasets.dataset import DatasetBuilder


class CustomDatasetBuilder(DatasetBuilder):

    def _read(self, filename: str, *args):
        file_mode = self.config.get('file_mode')
        assert file_mode, '`file_mode` must be specified'

        supported_modes = ['typical_mode', 'paddlenlp_mode']
        assert file_mode in supported_modes, f'Unsupported file mode, choose one from: {supported_modes}'
        assert os.path.isabs(filename), f'Absolute path needed, while `{filename}` is given'

        if file_mode == 'typical_mode':
            with open(filename, encoding='utf-8') as f:
                tokens = []
                tags = []
                for line in f:
                    if line == '\n':
                        if tokens:
                            yield {'tokens': tokens, 'labels': tags}
                        tokens = []
                        tags = []
                        continue
                    n, g = line.strip().split()
                    tokens.append(n)
                    tags.append(g)
        else:
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    line_stripped = line.strip().split('\t')
                    if not line_stripped:
                        break
                    if len(line_stripped) == 2:
                        tokens = line_stripped[0].split("\002")
                        tags = line_stripped[1].split("\002")
                    else:
                        tokens = line_stripped.split("\002")
                        tags = []
                    yield {"tokens": tokens, "labels": tags}

    def get_labels(self):
        """You should list all labels in your annotated data manually."""
        return ['B-DEVICE', 'I-DEVICE', 'B-LOCATION', 'I-LOCATION', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-OTHERS',
                'I-OTHERS', 'B-PERSON', 'I-PERSON', 'B-SYSTEM', 'I-SYSTEM', 'B-TECH', 'I-TECH', 'B-TIME', 'I-TIME',
                'O']

    def _get_data(self, mode: str):
        """Not used in this project"""
        raise NotImplementedError
