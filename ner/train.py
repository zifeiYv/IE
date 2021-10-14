# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import time
from functools import partial

import paddle
from paddle.io import DataLoader

from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer
from paddlenlp.data import Stack, Pad, Dict

from utils import CustomDatasetBuilder
cwd = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_file", default='.', help="Absolute path to a checkpoint file, specify it when you want to load pretained parameters.")
parser.add_argument("--train_data_file", required=True, help="File name of training data.")
parser.add_argument("--test_data_file", help="File name of training data.")
parser.add_argument("--output_dir", default="./checkpoint", help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.", )
parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"], help="The device to select to train the model, is must be cpu/gpu/xpu.")
parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
parser.add_argument("--save_steps", default=5000, type=int, help="Save model parameters every `save_steps` steps.")
parser.add_argument("--save_prefix", help="Prefix of names of to be saved models")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--file_mode", required=True, choices=('typical_mode', 'paddlenlp_mode'), help="Specify the type of annotated data.")


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader, label_num):
    model.eval()
    metric.reset()
    avg_loss, precision, recall, f1_score = 0, 0, 0, 0
    for batch in data_loader:
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            length, preds, labels)
        metric.update(num_infer_chunks.numpy(),
                      num_label_chunks.numpy(), num_correct_chunks.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval loss: %f" % avg_loss)
    model.train()
    return precision, recall, f1_score


def tokenize_and_align_labels(example, tokenizer, no_entity_id,
                              max_seq_len=512):
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


def do_train(args):
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # Load train dataset
    train_ds = CustomDatasetBuilder(lazy=False, file_mode=args.file_mode).read_datasets(
        data_files=os.path.join(cwd, 'custom_datasets', args.train_data_file))
    label_list = train_ds.label_list
    label_num = len(label_list)
    no_entity_id = label_num - 1

    # Load pretrained model ERNIE
    # You cannot load model from Internet since network connection is unavailable
    model = BertForTokenClassification.from_pretrained(os.path.join(cwd, 'models'), num_classes=label_num)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(cwd, 'tokenizer'))

    # Load model parameters if where has one
    if os.path.isfile(args.checkpoint_file):
        model.set_dict(paddle.load(args.checkpoint_file))

    # Load test dataset if passed
    has_test_data = False
    if args.test_data_file:
        assert os.path.isfile(os.path.join(cwd, 'custom_datasets', args.test_data_file))
        has_test_data = True
        test_ds = CustomDatasetBuilder(lazy=False, file_mode=args.file_mode).read_datasets(
            data_files=os.path.join(cwd, 'custom_datasets', args.test_data_file))

    ignore_label = -100

    trans_func = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        no_entity_id=no_entity_id,
        max_seq_len=args.max_seq_length)

    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32'),  # input
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32'),  # segment
        'seq_len': Stack(dtype='int64'),  # seq_len
        'labels': Pad(axis=0, pad_val=ignore_label, dtype='int64')  # label
    }): fn(samples)

    train_ds = train_ds.map(trans_func)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_data_loader = DataLoader(
        dataset=train_ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_sampler=train_batch_sampler,
        return_list=True)

    if has_test_data:
        test_ds = test_ds.map(trans_func)
        test_data_loader = DataLoader(
            dataset=test_ds,
            collate_fn=batchify_fn,
            num_workers=0,
            batch_size=args.batch_size,
            return_list=True)

    # Define learning rate strategy
    steps_by_epoch = len(train_data_loader)  # how many training steps in one epoch
    num_training_steps = steps_by_epoch * args.num_train_epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_ratio)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)
    loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
    metric = ChunkEvaluator(label_list=label_list)

    # Start training
    global_step = 0
    last_step = args.num_train_epochs * len(train_data_loader)
    tic_train = time.time()
    prefix = args.save_prefix if args.save_prefix else 'custom'
    for epoch in range(args.num_train_epochs):
        print(f"\n=====start training of {epoch+1} epochs=====")
        tic_epoch = time.time()
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, token_type_ids, _, labels = batch
            logits = model(input_ids, token_type_ids)
            loss = loss_fct(logits, labels)
            avg_loss = paddle.mean(loss)
            if global_step % args.logging_steps == 0 and rank == 0:
                print(f"epoch: {epoch+1} / {args.num_train_epochs}, steps: {step} / {steps_by_epoch}, "
                      f"loss: {avg_loss.numpy()[0]:.6f}, "
                      f"speed: {(args.logging_steps / (time.time() - tic_train)):.2f} step/s")
                tic_train = time.time()
            avg_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % args.save_steps == 0 and rank == 0:
                if has_test_data:
                    print(f"\n=====start evaluating ckpt of {global_step} steps=====")
                    precision, recall, f1 = evaluate(model, loss_fct, metric, test_data_loader, label_num)
                    print(f"precision: {precision * 100:.2f}\trecall: {recall * 100:.2f}\tf1: {f1 * 100:.2f}")
                paddle.save(model.state_dict(), os.path.join(args.output_dir, f"{prefix}_model_{global_step}.pdparams"))
        tic_epoch = time.time() - tic_epoch
        print(f"epoch time footprint: {tic_epoch // 3600} hour(s) {(tic_epoch % 3600) // 60} min(s)"
              f" {tic_epoch % 60:.2f} sec(s)")

    # Do the final evaluation
    if rank == 0 and has_test_data:
        print(f"\n=====start evaluating ckpt of {global_step} steps=====")
        precision, recall, f1 = evaluate(model, loss_fct, metric, test_data_loader, label_num)
        print(f"precision: {precision * 100:.2f}\trecall: {recall * 100:.2f}\tf1: {f1 * 100:.2f}")
    paddle.save(model.state_dict(), os.path.join(args.output_dir, f"{prefix}_model_{global_step}.pdparams"))
    print("\n=====training complete=====")


if __name__ == "__main__":
    arguments = parser.parse_args()
    do_train(arguments)
