# Parallel-Meaning-Bank-5

[![Logo](logo.png)](#)

## Table of Contents
1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Experiments](#experiments)
4. [Model Usage](#model-usage)
5. [Evaluation](#evaluation)
6. [License](#license)
7. [Contributors](#contributors)

## Overview

Data are available at /data folder, including pmb-5.0.0. pmb-5.0.1 and pmb 5.1.0.

Recommendation: please use the latest version: 5.1.0.

In this GitHub repository, we only provide the seq2seq data. For more detailed data, please refer to [PMB's official website](https://pmb.let.rug.nl/releases/).

## Dataset Description
Training a DRS (SBN) parser using PMB data involves utilizing gold, silver, bronze, and copper data. The statistics of pmb-5.1.0 are as follows:

|            | Gold-Train | Gold-Dev | Gold-Test | Silver | Bronze | Copper |
|------------|------------|----------|-----------|--------|--------|--------|
| English    | 9560       | 1195     | 1195      | 146718 | 141435 |        |
| German     | 1256       | 936      | 936       | 6946   |        | 155974 |
| Dutch      | 600        | 447      | 447       | 1660   |        | 29116  |
| Italian    | 776        | 576      | 576       | 4336   |        | 94648  |

## Experiments
The following table presents parsing results, please refer to https://pmb.let.rug.nl/models.php

| Parser   | English F1 | ERR | Dutch F1 | ERR | Italian F1 | ERR | German F1 | ERR |
|----------|------------|-----|----------|-----|------------|-----|-----------|-----|
| LSTM     | 78.6       | 8.4 | 80.2     | 4.0 | 74.4       | 8.5 | 79.6      | 5.0 |
| mT5      | 88.8       | 2.8 | 86.7     | 1.9 | 47.0       | 16.0| 82.0      | 2.8 |
| byT5     | 91.4       | 2.1 | 88.0     | 0.7 | 79.8       | 5.0 | 87.2      | 0.7 |
| mBART    | 89.1       | 2.3 | 86.1     | 1.8 | 64.5       | 3.4 | 86.2      | 1.8 |
| DRS-MLM  | 91.5       | 1.5 | 87.1     | 2.1 | 85.5       | 2.0 | 87.2      | 0.9 |

## Model Usage
To use the pre-trained mBart model by [Chunliu and Huiyuan](https://github.com/wangchunliu/DRS-pretrained-LMM), run:

```bash
python3 src/model/mBART/run.py
```

## Contributors
Xiao Zhang, Chunliu Wang, Rik van Noord, Johan Bos
