#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
This script aims to create train/dev/split for seq2seq models training and testing
In order to keep clean, we store the data in ../../data
"""

import argparse
import os
import re

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", '--split_path', default=os.path.join(path, "data/pmb-5.0.1/split"),
                        help="Root of release files")
    parser.add_argument("-s", '--save_path', default=os.path.join(path, "data/pmb-5.0.1/seq2seq"),
                        help="Path to save one-line-sbn files")
    args = parser.parse_args()
    return args

def data2dict(data_path):
    with open(data_path, "r", encoding="utf-8") as f1:
        data = f1.readlines()

    data_dict = {}
    for i in range(len(data)):
        d = data[i]
        if re.match('^p\d+/d\d+$', d):
            idx = d.strip()
            flag = True
            raw_sbn = []
        elif flag and d != "\n":
            raw_sbn.append(d)
        else:
            data_dict[idx] = raw_sbn
            flag = False

    return data_dict


def dict2data(data_dict, data_path):
    with open(data_path, "w", encoding="utf-8") as w:
        for key in data_dict.keys():
            try:
                raw = data_dict[key][0].strip().replace("\t", " ")
                sbn = data_dict[key][1].strip().replace("\t", " ")
            except Exception as e:
                print(f"{key} in {data_path} is sbn-empty!")
            w.write(raw + "\t")
            w.write(sbn + "\n")


if __name__ == "__main__":
    args = create_arg_parser()

    for root, dirs, files in os.walk(args.split_path):
        for file in files:
            if file.endswith('.sbn'):
                flag = False
                file_path = os.path.join(root, file)
                data_dict = data2dict(file_path)

                if not os.path.exists(root.replace("split", "seq2seq")):
                    os.makedirs(root.replace("split", "seq2seq"))

                dict2data(data_dict, os.path.join(root.replace("split", "seq2seq"), file))
