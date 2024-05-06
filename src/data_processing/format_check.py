#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Checking the format of sbn file
@File ：format_check.py
@Author ：xiao zhang
@Date ：2023/8/4 12:45 
'''



import argparse
import os
import re


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", '--file_path', default="G:\github\PMB5.0.0\data\pmb-5.0.0\seq2seq\en\\train\silver.sbn", help="Root of release files")
    args = parser.parse_args()
    return args


def check_space(text):
    tab_count = text.count("\t")
    if tab_count == 1:
        return True
    else:
        return False


if __name__ == "__main__":
    args = create_arg_parser()

    with open(args.file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if not check_space(line):
            print(f"----- {line} ----- HAS TWO TABS!!")
