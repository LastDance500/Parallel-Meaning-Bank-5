[![Logo](new_logo.png)](#)

# Parallel-Meaning-Bank-5

## Table of Contents
1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Model Training](#model-usage)
4. [Evaluation](#evaluation)
5. [Contributors](#contributors)

## Overview

Data are available at /data folder, including:

✅ pmb-5.0.0  
✅ pmb-5.0.1  
✅ pmb-5.1.0  
❌ pmb-4.0.0

👉 Recommendation: please use the latest version 5.1.0.

❗ In this GitHub repository, we only provide the seq2seq data. For more detailed data, please refer to [PMB's official website](https://pmb.let.rug.nl/releases/).

## Dataset Description

Training a DRS (SBN) parser using PMB data involves utilizing gold, silver, bronze, and copper data. The statistics of pmb-5.1.0 are as follows:

|            | Gold-Train🥇 | Gold-Dev🥇 | Gold-Test🥇 | Silver🥈 | Bronze🥉 | Copper🥉 |
|------------|------------|----------|-----------|--------|----------|----------|
| English🇬🇧    | 9560       | 1195     | 1195      | 146718 | 141435   | -        |
| German🇩🇪     | 1256       | 936      | 936       | 6946   | -        | 155974   |
| Dutch🇳🇱      | 600        | 447      | 447       | 1660   | -        | 29116    |
| Italian🇮🇹    | 776        | 576      | 576       | 4336   | -        | 94648    |


## Model Training

🤖 To change the hyperparameters, please go to src/parsing/model.py

```bash
python3 src/parsing/run.py -l en -m google/byt5-base -ip -pt data/pmb-5.1.0/seq2seq/en/train/gold_silver.sbn -t data/pmb-5.1.0/seq2seq/en/train/gold.sbn -d data/pmb-5.1.0/seq2seq/en/dev/standard.sbn -e data/pmb-5.1.0/seq2seq/en/test/standard.sbn -c data/pmb-5.1.0/seq2seq/en/test/long.sbn -s results/parsing/ -epoch 50 -lr 1e-4
```

## Contributors

🐄 Xiao Zhang, Chunliu Wang, Rik van Noord, Johan Bos
