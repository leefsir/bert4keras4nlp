# _*_coding:utf-8_*_
# author leewfeng
# 2021/1/4 19:25
# 基本测试：中文GPT2_ML模型
# 介绍链接：https://kexue.fm/archives/7292


import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout


