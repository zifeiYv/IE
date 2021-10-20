# -*- coding: utf-8 -*-
import json
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader

from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification

from ere.data_loader import DuIEDataset, DataCollator, ChineseAndPunctuationExtractor, convert_example_to_feature
from ere.utils import decoding
from config import device

# >>> User defined content required
paddle.set_device(device)
self_model = 'ere_model_for_general_domain.pdparams'
# <<< Finished

cwd = os.getcwd()


class BCELossForDuIE(nn.Layer):
    def __init__(self, ):
        super(BCELossForDuIE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        mask = paddle.cast(mask, 'float32')
        loss = loss * mask.unsqueeze(-1)
        loss = paddle.sum(loss.mean(axis=2), axis=1) / paddle.sum(mask, axis=1)
        loss = loss.mean()
        return loss


with open(os.path.join(cwd, 'ere/config_data/predicate2id.json'), encoding='utf8') as fp:
    label_map = json.load(fp)
num_classes = (len(label_map.keys()) - 2) * 2 + 2

# Loads pretrained model ERNIE
model = ErnieForTokenClassification.from_pretrained(os.path.join(cwd, 'ere/models'), num_classes=num_classes)
tokenizer = ErnieTokenizer.from_pretrained(os.path.join(cwd, 'ere/tokenizer'))
criterion = BCELossForDuIE()
state_dict = paddle.load(os.path.join(cwd, 'ere/checkpoint/' + self_model))
model.set_dict(state_dict)


def predict(sentence):
    predict_dataset, example_all = read(sentence)
    collator = DataCollator()
    predict_batch_sampler = paddle.io.BatchSampler(predict_dataset, batch_size=128, shuffle=False)
    predict_data_loader = DataLoader(
        dataset=predict_dataset,
        batch_sampler=predict_batch_sampler,
        collate_fn=collator,
        return_list=True)
    loss_all = 0
    formatted_outputs = []
    current_idx = 0
    with open(os.path.join(cwd, 'ere/config_data/id2spo.json'), encoding='utf8') as fp:
        id2spo = json.load(fp)
    for batch in predict_data_loader:
        input_ids, seq_len, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
        logits = model(input_ids=input_ids)
        mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and((input_ids != 2))
        loss = criterion(logits, labels, mask)
        loss_all += loss.numpy().item()
        probs = F.sigmoid(logits)
        logits_batch = probs.numpy()
        seq_len_batch = seq_len.numpy()
        tok_to_orig_start_index_batch = tok_to_orig_start_index.numpy()
        tok_to_orig_end_index_batch = tok_to_orig_end_index.numpy()
        formatted_outputs.extend(decoding(example_all[current_idx: current_idx + len(logits)],
                                          id2spo,
                                          logits_batch,
                                          seq_len_batch,
                                          tok_to_orig_start_index_batch,
                                          tok_to_orig_end_index_batch))
        current_idx = current_idx + len(logits)
    res = []
    for formatted_instance in formatted_outputs:
        res.append(formatted_instance)
    return res


def read(sentence):
    if isinstance(sentence, str):
        sentences = [{'text': sentence}]
    else:
        if isinstance(sentence, list):
            sentences = []
            for line in sentence:
                sentences.append({'text': line})
        else:
            raise
    chineseandpunctuationextractor = ChineseAndPunctuationExtractor()
    input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels = ([] for _ in range(5))
    for line in sentences:
        line = line['text']
        line = line.strip()
        input_feature = convert_example_to_feature(
            {'text': line}, tokenizer, chineseandpunctuationextractor, label_map, 128, True)
        input_ids.append(input_feature.input_ids)
        seq_lens.append(input_feature.seq_len)
        tok_to_orig_start_index.append(input_feature.tok_to_orig_start_index)
        tok_to_orig_end_index.append(input_feature.tok_to_orig_end_index)
        labels.append(input_feature.labels)
    return DuIEDataset(input_ids, seq_lens, tok_to_orig_start_index,
                       tok_to_orig_end_index, labels), sentences

