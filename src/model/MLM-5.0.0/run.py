#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DRG_parsing 
@File ：run.py
@Author ：xiao zhang
@Date ：2022/11/14 12:27
'''

import argparse
import os
import sys
sys.path.append(".")

from model import get_dataloader, Generator

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lang", required=False, type=str, default="it",
                        help="language in [en, nl, de ,it]")
    parser.add_argument("-pt", "--pretrain", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.0.0/seq2seq/it/train/gold_silver_copper.sbn"),
                        help="text input file")
    parser.add_argument("-t", "--train", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.0.0/seq2seq/it/train/gold.sbn"),
                        help="text input file")
    parser.add_argument("-dti", "--dev", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.0.0/seq2seq/it/dev/standard.sbn"),
                        help="dev text input file")
    parser.add_argument("-tti", "--test", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.0.0/seq2seq/it/test/standard.sbn"),
                        help="test text input file")
    parser.add_argument("-tti2", "--test2", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.0.0/seq2seq/it/test/long.sbn"),
                        help="test text input file")
    parser.add_argument("-s", "--save1", required=False, type=str,
                        default=os.path.join(path, "src/model/MLM-5.0.0/result/MLM5_it_standard_.txt"),
                        help="path to save the result")
    parser.add_argument("-s2", "--save2", required=False, type=str,
                        default=os.path.join(path, "src/model/MLM-5.0.0/result/MLM5_it_long.txt"),
                        help="path to save the second result")
    parser.add_argument("-tl", "--test_long", required=False, type=str,
                           default="false",
                        help="path to save the second result")
    parser.add_argument("-m", "--mode", required=False, type=str,
                        default="train",
                        help="train or test")
    parser.add_argument("-ta", "--target", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.0.0/seq2seq/it/dev/standard.sbn"),
                        help="train or test")
    parser.add_argument("-s3", "--save3", required=False, type=str,
                        default=os.path.join(path, "src/model/DRS-MLM/result/MLM_it_standard.txt.dev"),
                        help="train or test")
    args = parser.parse_args()

    return args


def main():
    args = create_arg_parser()

    # train process
    lang = args.lang
    train_dataloader1 = get_dataloader(args.pretrain)
    train_dataloader2 = get_dataloader(args.train)
    test_dataloader = get_dataloader(args.test)
    dev_dataloader = get_dataloader(args.dev)

    save_path1 = args.save1
    save_path2 = args.save2
    save_path3 = args.save3

    print(lang)

    mode = args.mode

    test_long = args.test_long

    if mode == "train":
        if test_long == "true":
            test_dataloader2 = get_dataloader(args.test2)
            bart_classifier = Generator(lang)
            bart_classifier.train(train_dataloader1, test_dataloader, lr=0.0001, epoch_number=3)
            bart_classifier.train(train_dataloader2, test_dataloader, lr=0.0001, epoch_number=5)
            bart_classifier.evaluate(test_dataloader, save_path1)
            bart_classifier.evaluate(test_dataloader2, save_path2)
        else:
            bart_classifier = Generator(lang)
            bart_classifier.train(train_dataloader1, test_dataloader, lr=0.0001, epoch_number=3)
            bart_classifier.train(train_dataloader2, test_dataloader, lr=0.0001, epoch_number=5)
            bart_classifier.evaluate(dev_dataloader, save_path1+".dev")
            bart_classifier.evaluate(test_dataloader, save_path1)

        bart_classifier.model.save_pretrained(os.path.join(path, f"models/MLM5_seq2seq/{lang}"))
    else:
        test_dataloader3 = get_dataloader(args.target)
        bart_classifier = Generator(lang, load_path=os.path.join(path, f"models/MLM5_seq2seq/{lang}"))
        bart_classifier.evaluate(test_dataloader3, save_path3)


if __name__ == '__main__':
    main()
