# -*- coding: utf-8 -*-
from functools import partial
import warnings
import os

import paddle
from paddle.io import DataLoader

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Pad, Dict
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer

# >>> User defined content required
paddle.set_device('cpu')
label_list = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]
self_model = 'ner_model_for_general_domain.pdparams'
# <<< Finished

warnings.filterwarnings('ignore')
cwd = os.getcwd()


def tokenize_and_align_labels(example, tokenizer, no_entity_id, max_seq_len=512):
    labels = example['labels']
    example = example['tokens']
    tokenized_input = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (
            len(tokenized_input['input_ids']) - len(tokenized_input['labels']))

    return tokenized_input


def parse_decodes(input_words, id2label, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]

    outputs = []
    for idx, end in enumerate(lens):
        sent = "".join(input_words[idx]['tokens'])
        tags = [id2label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.startswith('B-') or t == 'O':
                if len(words):
                    sent_out.append(words)
                if t.startswith('B-'):
                    tags_out.append(t.split('-')[1])
                else:
                    tags_out.append(t)
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs


label_num = len(label_list)
no_entity_id = label_num - 1

tokenizer = BertTokenizer.from_pretrained(os.path.join(cwd, 'ner/tokenizer'))

trans_func = partial(
    tokenize_and_align_labels,
    tokenizer=tokenizer,
    no_entity_id=no_entity_id,
    max_seq_len=128)

ignore_label = -100
fn = Dict({
    'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    'seq_len': Stack(),
    'labels': Pad(axis=0, pad_val=ignore_label)  # label
})


def batchify_fn(samples, func):
    return func(samples)

# batchify_fn = lambda samples, fn=Dict({
#     'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
#     'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
#     'seq_len': Stack(),
#     'labels': Pad(axis=0, pad_val=ignore_label)  # label
# }): fn(samples)


# Define the model network
model = BertForTokenClassification.from_pretrained(os.path.join(cwd, 'ner/models'), num_classes=label_num)

model_dict = paddle.load(os.path.join(cwd, 'ner/checkpoint/' + self_model))
model.set_dict(model_dict)
model.eval()


def predict(sentence):
    predict_ds = load_dataset(
        read,
        sentence=sentence,
        lazy=False)
    predict_ds.label_list = label_list
    raw_data = predict_ds.data
    id2label = dict(enumerate(predict_ds.label_list))
    predict_ds = predict_ds.map(trans_func)
    predict_data_loader = DataLoader(
        dataset=predict_ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_size=128,
        return_list=True)
    pred_list = []
    len_list = []
    for step, batch in enumerate(predict_data_loader):
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(length.numpy())
    preds = parse_decodes(raw_data, id2label, pred_list, len_list)
    return preds


def read(sentence):
    if isinstance(sentence, str):
        sentences = [sentence]
    else:
        if isinstance(sentence, list):
            sentences = sentence
        else:
            raise
    for line in sentences:
        line = line.strip()
        words = list(line)
        yield {'tokens': words, 'labels': []}
