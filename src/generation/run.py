#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：PMB5
@File ：run.py
@Author ：xiao zhang
@Date ：2022/11/14 12:27
'''

import argparse
import os
import sys
sys.path.append(".")

from model import get_dataloader, Generator

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lang", required=False, type=str, default="en",
                        help="language in [en, nl, de ,it]")
    parser.add_argument("-m", "--model", required=False, type=str, default="google/byt5-base",
                        help="The model you want to use, better choose from byT5, mbart, mT5")
    parser.add_argument("-ip", "--if_pre",
                        action='store_true',
                        help="if pre-train or not, here pre-train means fine-tuning on silver")
    parser.add_argument("-pt", "--pretrain", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.1.0/seq2seq/en/train/gold_silver.sbn"),
                        help="pre-train (fine-tuning) sets")
    parser.add_argument("-t", "--train", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.1.0/seq2seq/en/train/gold.sbn"),
                        help="train sets")
    parser.add_argument("-d", "--dev", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.1.0/seq2seq/en/dev/standard.sbn"),
                        help="dev sets")
    parser.add_argument("-e", "--test", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.1.0/seq2seq/en/test/standard.sbn"),
                        help="standard test sets")
    parser.add_argument("-c", "--challenge", nargs='*', required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.1.0/seq2seq/en/test/long.sbn"),
                        help="challenge sets you also want to test")
    parser.add_argument("-s", "--save", required=False, type=str,
                        default=os.path.join(path, "results/parsing/language/model_name"),
                        help="path to save the result file")
    parser.add_argument("-epoch", "--epoch", required=False, type=int,
                        default=10)
    parser.add_argument("-lr", "--learning_rate", required=False, type=float,
                        default=1e-05)
    parser.add_argument("-ms", "--model_save", required=False, type=str,
                        default=os.path.join(path, "trained_model/model_name"))
    args = parser.parse_args()
    return args


def main():
    args = create_arg_parser()

    # train process
    lang = args.lang
    model = args.model

    # train loader
    train_dataloader = get_dataloader(args.train)

    # test loader
    test_dataloader = get_dataloader(args.test)
    dev_dataloader = get_dataloader(args.dev)

    # save path
    save_path = args.save

    # hyperparameters
    epoch = args.epoch
    lr = args.learning_rate

    # train
    bart_classifier = Generator(lang, model)

    if args.if_pre: # if pretrain or not
        train_dataloader_pre = get_dataloader(args.pretrain)
        bart_classifier.train(train_dataloader_pre, dev_dataloader, lr=lr, epoch_number=3)
    bart_classifier.train(train_dataloader, dev_dataloader, lr=lr, epoch_number=epoch)

    # Define the full path to the final directory
    final_path = os.path.join(save_path, lang, model.replace('/', '-'))

    # Create the directories in one go
    os.makedirs(final_path, exist_ok=True)

    # standard test
    bart_classifier.evaluate(test_dataloader, os.path.join(save_path, f"{lang}/{model.replace('/', '-')}/standard.sbn"))

    # challenge test
    for i in range(len(args.challenge)):
        cha_path = args.challenge[i]
        cha_dataloader = get_dataloader(cha_path)
        bart_classifier.evaluate(cha_dataloader, os.path.join(save_path,
                                                              f"{lang}/{model.replace('/', '-')}/challenge{i}.sbn"))

    # bart_classifier.model.save_pretrained(args.model_save)


if __name__ == '__main__':
    main()
