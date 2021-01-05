# _*_coding:utf-8_*_
# author leewfeng
# 2021/1/4 19:25
# 基本测试：中文GPT2_ML模型
# 介绍链接：https://kexue.fm/archives/7292


import os

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.tokenizers import Tokenizer

from basis_framework.basis_graph import BasisGraph
from configs.path_config import GPT2_MODEL_PATH
from utils.common_tools import load_json, save_json


class ExtractFeature(BasisGraph):
    def __init__(self, params={}, Train=False):
        super().__init__(params, Train)
        self.config_path = os.path.join(GPT2_MODEL_PATH + "/config.json")
        self.checkpoint_path = os.path.join(GPT2_MODEL_PATH + "/model.ckpt-100000")
        self.vocab_path = os.path.join(GPT2_MODEL_PATH + "/vocab.txt")
        self.tokenizer = Tokenizer(self.vocab_path, token_start=None, token_end=None, do_lower_case=True)

    def save_params(self):
        self.params['max_len'] = self.max_len
        save_json(jsons=self.params, json_path=self.params_path)

    def load_params(self):
        load_params = load_json(self.params_path)
        self.max_len = load_params.get('max_len')

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
        model = build_transformer_model(self.config_path, self.checkpoint_path, model='gpt2_ml')

        class ArticleCompletion(AutoRegressiveDecoder):
            """
            基于随机采样的文章续写
            """

            def __init__(self, start_id, end_id, maxlen, minlen=None, model=None, tokenizer=None):
                self.tokenizer = tokenizer
                super().__init__(start_id, end_id, maxlen, minlen=None)

            @AutoRegressiveDecoder.wraps(default_rtype='probas')
            def predict(self, inputs, output_ids, step):
                token_ids = np.concatenate([inputs[0], output_ids], 1)
                return self.last_token(model).predict(token_ids)

            def generate(self, text, n=1, topp=0.95):
                token_ids, _ = self.tokenizer.encode(text)
                results = self.random_sample([token_ids], n, topp=topp)
                return [text + self.tokenizer.decode(ids) for ids in results]

        self.article_completion = ArticleCompletion(start_id=None,
                                                    end_id=511,  # 511是中文句号
                                                    maxlen=256,
                                                    minlen=128)

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
        print(self.article_completion.generate(u'今天天气不错'))

    def save_model(self, model_path='test.model'):
        self.model.save(model_path)
        del self.model  # 释放内存

    def load_model(self, model_path='test.model'):
        # self.model = keras.models.load_model(model_path)
        # self.extract_features('语言模型')
        pass

    def load_params(self):
        pass
