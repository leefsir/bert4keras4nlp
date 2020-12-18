#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 15:13 
# ide： PyCharm

from __future__ import print_function, division

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from keras.layers import Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam

from basis_framework.basis_graph import BasisGraph
from configs.path_config import CORPUS_ROOT_PATH
from utils.classifier_data_process import Data_Generator, Evaluator
from utils.common_tools import data2csv, data_preprocess, split


class BertGraph(BasisGraph):
    def __init__(self, params={}, Train=False):
        if not params.get('model_code'):
            params['model_code'] = 'classifier'
        super().__init__(params, Train)

    def data_process(self, sep='\t'):
        """
        数据处理
        :return:
        """
        if '.csv' not in self.train_data_path:
            self.train_data_path = data2csv(self.train_data_path, sep)
        self.index2label, self.label2index, self.labels, train_data = data_preprocess(self.train_data_path)
        self.num_classes = len(self.index2label)
        if self.valid_data_path:
            if '.csv' not in self.valid_data_path:
                self.valid_data_path = data2csv(self.valid_data_path, sep)
            _, _, _, valid_data = data_preprocess(self.valid_data_path)
        else:
            train_data, valid_data = split(train_data, self.split)
        if self.test_data_path:
            if '.csv' not in self.test_data_path:
                self.test_data_path = data2csv(self.test_data_path, sep)
            _, _, _, test_data = data_preprocess(self.test_data_path)
        else:
            test_data = []
        self.train_generator = Data_Generator(train_data, self.label2index, self.tokenizer, self.batch_size,
                                                     self.max_len)
        self.valid_generator = Data_Generator(valid_data, self.label2index, self.tokenizer, self.batch_size,
                                                     self.max_len)
        self.test_generator = Data_Generator(test_data, self.label2index, self.tokenizer, self.batch_size,
                                                    self.max_len)

    def build_model(self):
        bert = build_transformer_model(
            config_path=self.bert_config_path,
            checkpoint_path=self.bert_checkpoint_path,
            return_keras_model=False,
        )
        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)  # 取出[cls]层对应的向量来做分类
        output = Dense(self.num_classes, activation=self.activation, kernel_initializer=bert.initializer)(
            output)  # 全连接层激活函数分类
        self.model = Model(bert.model.input, output)
        print(self.model.summary(150))

    def predict(self, text):
        token_ids, segment_ids = self.tokenizer.encode(text)
        pre = self.model.predict([[token_ids], [segment_ids]])
        res = self.index2label.get(str(np.argmax(pre[0])))
        return res

    def compile_model(self):
        # 派生为带分段线性学习率的优化器。
        # 其中name参数可选，但最好填入，以区分不同的派生优化器。
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model.compile(loss=self.loss,
                           optimizer=AdamLR(lr=self.learning_rate, lr_schedule={
                               1000: 1,
                               2000: 0.1
                           }),
                           metrics=self.metrics, )

    def train(self):
        # 保存超参数
        evaluator = Evaluator(self.model, self.model_path, self.valid_generator, self.test_generator)

        # 模型训练
        self.model.fit_generator(
            self.train_generator.forfit(),
            steps_per_epoch=len(self.train_generator),
            epochs=self.epoch,
            callbacks=[evaluator],
        )


if __name__ == '__main__':
    params = {
        'model_code': 'thuc_news_bert',
        'train_data_path': CORPUS_ROOT_PATH + '/thuc_news/train.txt',
        'valid_data_path': CORPUS_ROOT_PATH + '/thuc_news/dev.txt',
        'test_data_path': CORPUS_ROOT_PATH + '/thuc_news/test.txt',
        'batch_size': 128,
        'max_len': 30,
        'epoch': 10,
        'learning_rate': 1e-5,
        'gpu_id': 1,
    }
    bertModel = BertGraph(params, Train=True)
    bertModel.train()
else:
    params = {
        'model_code': 'thuc_news_bert',  # 此处与训练时code保持一致
        'gpu_id': 1,
    }
    bertModel = BertGraph(params)
