#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 14:10 
# ide： PyCharm
import os
# 项目的根目录
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
path_root = path_root.replace('\\', '/')
# print(path_root)
# 日志路径配置
LOG_PATH = os.path.join(path_root,"logs")
LOG_NAME = "classification.log"

# 模型文件路径
MODEL_ROOT_PATH = os.path.join(path_root,'models')
BERT_MODEL_PATH = os.path.join(MODEL_ROOT_PATH,'chinese_L-12_H-768_A-12')
GPT2_MODEL_PATH =os.path.join(MODEL_ROOT_PATH,'gpt2_ml')

# 训练语料路径
CORPUS_ROOT_PATH = os.path.join(path_root,'corpus')

# 实体字典路径
ENTITY_DICT = os.path.join(path_root,'/utils/dynamic_data_cache/entity.csv')

