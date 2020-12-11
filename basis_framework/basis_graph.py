#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/11 17:17
# ide： PyCharm
import os

from bert4keras.tokenizers import Tokenizer

from configs.path_config import MODEL_ROOT_PATH, BERT_MODEL_PATH
from utils.common_tools import load_json, save_json


class BasisGraph():
    def __init__(self, params={}, Train=False):
        self.bert_config_path = os.path.join(BERT_MODEL_PATH, 'bert_config.json')
        self.bert_checkpoint_path = os.path.join(BERT_MODEL_PATH, 'bert_model.ckpt')
        self.bert_vocab_path = os.path.join(BERT_MODEL_PATH, 'vocab.txt')
        self.tokenizer = Tokenizer(self.bert_vocab_path, do_lower_case=True)
        self.model_code = params.get('model_code')
        if not self.model_code: raise Exception("No model code!,params must have a 'model_code'")
        model_dir = os.path.join(MODEL_ROOT_PATH, self.model_code)
        if not os.path.exists(model_dir):os.makedirs(model_dir, exist_ok=True)
        self.params_path = os.path.join(model_dir, 'params.json')
        self.model_path = os.path.join(model_dir, 'best_model.weights')
        self.tensorboard_path = os.path.join(model_dir, 'logs')
        self.max_len = params.get('max_len', 128)
        self.batch_size = params.get('batch_size', 32)
        self.train_data_path = params.get('train_data_path')
        if Train and not self.train_data_path: raise Exception("No training data!")
        self.valid_data_path = params.get('valid_data_path')
        self.test_data_path = params.get('test_data_path')
        self.epoch = params.get('epoch', 10)
        self.learning_rate = params.get('learning_rate', 1e-5)  # bert_layers越小，学习率应该要越大
        self.bert_layers = params.get('bert_layers', 12)
        self.crf_lr_multiplier = params.get('crf_lr_multiplier', 1000)  # 必要时扩大CRF层的学习率
        self.gpu_id = params.get("gpu_id", None)
        self.activation = params.get('activation', 'softmax')  # 分类激活函数,softmax或者signod
        self.loss = params.get('loss','sparse_categorical_crossentropy')
        self.metrics = params.get('metrics',['accuracy'])
        self.split = params.get('split',0.8)  # 训练/验证集划分
        self.params = params
        self._set_gpu_id()  # 设置训练的GPU_ID
        if Train:
            self.data_process()
            self.save_params()
            self.build_model()
            self.compile_model()
        else:
            self.load_params()
            self.build_model()
            self.load_model()
    def save_params(self):
        self.params['num_classes'] = self.num_classes
        self.params['index2label'] = self.index2label
        self.params['max_len'] = self.max_len
        save_json(jsons=self.params, json_path=self.params_path)
    def load_params(self):
        load_params = load_json(self.params_path)
        self.max_len = load_params.get('max_len')
        self.num_classes = load_params.get('num_classes')
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
        raise NotImplementedError
    def model_create(self):
        """
        模型框架搭建
        :return:
        """
        raise NotImplementedError
    def compile_model(self):
        """
        模型框架搭建
        :return:
        """
        raise NotImplementedError
    def train(self):
        """
        模型框架搭建
        :return:
        """
        raise NotImplementedError
    def predict(self,text):
        """
        模型框架搭建
        :return:
        """
        raise NotImplementedError
    def load_model(self):
        self.model.load_weights(self.model_path)