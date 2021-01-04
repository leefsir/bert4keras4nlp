#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2021/1/4 11:33 
# ide： PyCharm
import os

from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.snippets import to_array

from basis_framework.basis_graph import BasisGraph
from utils.common_tools import load_json, save_json


class ExtractFeature(BasisGraph):
    def __init__(self, params={}, Train=False):
        super().__init__(params, Train)

    def save_params(self):
        self.params['num_classes'] = self.num_classes
        self.params['labels'] = self.labels
        self.params['index2label'] = self.index2label
        self.params['label2index'] = self.label2index
        self.params['max_len'] = self.max_len
        save_json(jsons=self.params, json_path=self.params_path)

    def load_params(self):
        load_params = load_json(self.params_path)
        self.max_len = load_params.get('max_len')
        self.labels = load_params.get('labels')
        self.num_classes = load_params.get('num_classes')
        self.label2index = load_params.get('label2index')
        self.index2label = load_params.get('index2label')

    def _set_gpu_id(self):
        """指定使用的GPU显卡id"""
        if self.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

    def data_process(self):
        """
        模型框架搭建
        :return:
        """
        raise NotImplementedError

    def build_model(self):
        """
        模型框架搭建
        :return:
        """
        self.model = build_transformer_model(self.bert_config_path, self.bert_checkpoint_path)

    def compile_model(self):
        """
        模型框架搭建
        :return:
        """
        raise NotImplementedError

    def extract_features(self, text: str):
        """
        编码测试
        :return:
        """
        token_ids, segment_ids = self.tokenizer.encode(u'{}'.format(text))
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        print("\n === features === \n")
        print(self.predict([token_ids, segment_ids]))

    def save_model(self, model_path='test.model'):
        self.model.save(model_path)
        del self.model  # 释放内存

    def load_model(self, model_path='test.model'):
        # self.model = keras.models.load_model(model_path)
        # self.extract_features('语言模型')
        pass

    def load_params(self):
        pass