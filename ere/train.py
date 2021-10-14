# -*- coding: utf-8 -*-
# Author: sunjw
import argparse
import os
import time
import json
from tqdm import tqdm

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader

from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification, LinearDecayWithWarmup

from data_loader import DuIEDataset, DataCollator
from utils import decoding, get_precision_recall_f1, write_prediction_results

cwd = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_file", default='.', help="Absolute path to a checkpoint file, specify it when you want to load pretained parameters.")
parser.add_argument("--train_data_file", required=True, help="File name of training data.")
parser.add_argument("--test_data_file", help="File name of training data.")
parser.add_argument("--output_dir", default="./checkpoint", help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--logging_steps", default=100, type=int, help="Log progress message every `logging_steps` steps.")
parser.add_argument("--save_steps", default=5000, type=int, help="Save model parameters every `save_steps` steps.")
parser.add_argument("--save_prefix", help="Prefix of names of to be saved models")


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


@paddle.no_grad()
def evaluate(model, criterion, data_loader, file_path):
    """
    mode eval:
    eval on development set and compute P/R/F1, called between training.
    mode predict:
    eval on development / test set, then write predictions to \
        predict_test.json and predict_test.json.zip \
        under args.data_path dir for later submission or evaluation.
    """
    example_all = []
    with open(file_path, "r", encoding="utf-8") as fp:
        for line in fp:
            example_all.append(json.loads(line))
    id2spo_path = os.path.join(os.path.dirname(file_path), "id2spo.json")
    with open(id2spo_path, 'r', encoding='utf8') as fp:
        id2spo = json.load(fp)

    model.eval()
    loss_all = 0
    eval_steps = 0
    formatted_outputs = []
    current_idx = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        eval_steps += 1
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
    loss_avg = loss_all / eval_steps
    print("eval loss: %f" % loss_avg)

    predict_file_path = os.path.join(cwd, 'custom_datasets', 'predict_eval.json')

    predict_zipfile_path = write_prediction_results(formatted_outputs, predict_file_path)

    precision, recall, f1 = get_precision_recall_f1(file_path, predict_zipfile_path)
    os.system('rm {} {}'.format(predict_file_path, predict_zipfile_path))
    return precision, recall, f1


def do_train(args):
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    label_map_path = os.path.join(cwd, "config_data/predicate2id.json")
    if not os.path.isfile(label_map_path):
        raise FileExistsError(f"{label_map_path} dose not exists or is not a file.")
    with open(label_map_path, encoding='utf-8') as f:
        label_map = json.load(f)
    num_classes = (len(label_map.keys()) - 2) * 2 + 2

    # Loads pretrained model ERNIE
    # You cannot load model from Internet since network connection is unavailable
    model = ErnieForTokenClassification.from_pretrained(os.path.join(cwd, 'models'), num_classes=num_classes)
    model = paddle.DataParallel(model)
    tokenizer = ErnieTokenizer.from_pretrained(os.path.join(cwd, 'tokenizer'))
    criterion = BCELossForDuIE()
    # Load model parameters if there has one
    if os.path.isfile(args.checkpoint_file):
        model.set_dict(paddle.load(args.checkpoint_file))

    # Loads dataset
    train_dataset = DuIEDataset.from_file(os.path.join(cwd, 'custom_datasets', args.train_data_file),
                                          tokenizer, args.max_seq_length, True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, args.batch_size,
                                                            shuffle=True)
    collator = DataCollator()
    train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                   collate_fn=collator, return_list=True)

    has_test_data = False
    if args.test_data_file:
        assert os.path.isfile(os.path.join(cwd, 'custom_datasets', args.test_data_file))
        has_test_data = True
        test_dataset = DuIEDataset.from_file(os.path.join(cwd, 'custom_datasets', args.test_data_file),
                                             tokenizer, args.max_seq_length, True)
        test_batch_sampler = paddle.io.BatchSampler(test_dataset, args.batch_size,
                                                    shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler,
                                      collate_fn=collator, return_list=True)

    # Define learning rate strategy
    steps_by_epoch = len(train_data_loader)  # how many training steps in one epoch
    num_training_steps = steps_by_epoch * args.num_train_epochs  # total training steps
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_ratio)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ['bias', 'norm'])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params
    )

    # Start training
    global_step = 0
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    tic_train = time.time()
    prefix = args.save_prefix if args.save_prefix else 'custom'
    for epoch in range(args.num_train_epochs):
        print("\n=====start training of %d epochs=====" % epoch)
        tic_epoch = time.time()
        model.train()
        for step, batch in enumerate(train_data_loader):
            input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
            logits = model(input_ids=input_ids)
            mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and((input_ids != 2))
            loss = criterion(logits, labels, mask)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            loss_item = loss.numpy().item()
            global_step += 1

            if global_step % logging_steps == 0 and rank == 0:
                print(f"epoch: {epoch} / {args.num_train_epochs}, steps: {step} / {steps_by_epoch}, "
                      f"loss: {loss_item}, speed: {(logging_steps / (time.time() - tic_train)):.2f} step/s")
                tic_train = time.time()

            if global_step % save_steps == 0 and rank == 0:
                if has_test_data:
                    print(f"\n=====start evaluating ckpt of {global_step} steps=====")
                    precision, recall, f1 = evaluate(model, criterion, test_data_loader,
                                                     os.path.join(cwd, 'custom_datasets', args.test_data_file))
                    print(f"precision: {precision*100:.2f}\trecall: {recall*100:.2f}\tf1: {f1*100:.2f}")
                print(f"saving checkpoint model_{global_step}.pdparams to {args.output_dir}")
                prefix = args.save_prefix if args.save_prefix else 'custom'
                paddle.save(model.state_dict(), os.path.join(args.output_dir, f"{prefix}_model_{global_step}.pdparams"))
                model.train()
        tic_epoch = time.time() - tic_epoch
        print(f"epoch time footprint: {tic_epoch // 3600} hour(s) {(tic_epoch % 3600) // 60} min(s)"
              f" {tic_epoch % 60:.2f} sec(s)")

    # Do the final evaluation
    if rank == 0 and has_test_data:
        print(f"\n=====start evaluating ckpt of {global_step} steps=====")
        precision, recall, f1 = evaluate(model, criterion, test_data_loader,
                                         os.path.join(cwd, 'custom_datasets', args.test_data_file))
        print(f"precision: {precision * 100:.2f}\trecall: {recall * 100:.2f}\tf1: {f1 * 100:.2f}")
    paddle.save(model.state_dict(), os.path.join(args.output_dir, f"{prefix}_model_{global_step}.pdparams"))
    print("\n=====training complete=====")


if __name__ == '__main__':
    arguments = parser.parse_args()
    do_train(arguments)
