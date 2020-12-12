# _*_coding:utf-8_*_
# author leewfeng
# 2020/12/12 20:05
import json

import keras
import numpy as np
from bert4keras.snippets import DataGenerator, sequence_padding
from tqdm import tqdm

from utils.common_tools import search


def data_process(train_data_file_path, valid_data_file_path, max_len, params_path):
    train_data = json.load(open(train_data_file_path, encoding='utf-8'))

    if valid_data_file_path:
        train_data_ret = train_data
        valid_data_ret = json.load(open(valid_data_file_path, encoding='utf-8'))
    else:
        split = int(len(train_data) * 0.8)
        train_data_ret, valid_data_ret = train_data[:split], train_data[split:]
    p2s_dict = {}
    p2o_dict = {}
    predicate = []

    for content in train_data:
        for spo in content.get('new_spo_list'):
            s_type = spo.get('s').get('type')
            p_key = spo.get('p').get('entity')
            o_type = spo.get('o').get('type')
            if p_key not in p2s_dict:
                p2s_dict[p_key] = s_type
            if p_key not in p2o_dict:
                p2o_dict[p_key] = o_type
            if p_key not in predicate:
                predicate.append(p_key)
    i2p_dict = {i: key for i, key in enumerate(predicate)}
    p2i_dict = {key: i for i, key in enumerate(predicate)}
    save_params = {}
    save_params['p2s_dict'] = p2s_dict
    save_params['i2p_dict'] = i2p_dict
    save_params['p2o_dict'] = p2o_dict
    save_params['maxlen'] = max_len
    save_params['num_classes'] = len(i2p_dict)
    # 数据保存
    json.dump(save_params,
              open(params_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    return train_data_ret, valid_data_ret, p2s_dict, p2o_dict, i2p_dict, p2i_dict


class Data_Generator(DataGenerator):
    """数据生成器
    """

    def __init__(self, data, batch_size, tokenizer, p2i_dict, maxlen):
        super().__init__(data, batch_size=batch_size)
        self.tokenizer = tokenizer
        self.p2i_dict = p2i_dict
        self.maxlen = maxlen

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(d['text'], max_length=self.maxlen)
            # 整理三元组 {s: [(o_start,0_end, p)]}/{s_token_ids:[]}
            spoes = {}
            for spo in d['new_spo_list']:
                s = spo['s']
                p = spo['p']
                o = spo['o']
                s_token = self.tokenizer.encode(s['entity'])[0][1:-1]
                p = self.p2i_dict[p['entity']]
                o_token = self.tokenizer.encode(o['entity'])[0][1:-1]
                s_idx = search(s_token, token_ids)  # s_idx s起始位置
                o_idx = search(o_token, token_ids)  # o_idx o起始位置
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s_token) - 1)  # s s起始结束位置，s的类别
                    o = (o_idx, o_idx + len(o_token) - 1, p)  # o o起始结束位置及p的id,o的类别
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签，采用二维向量分别标记subject的起始位置和结束位置
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(self.p2i_dict), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(batch_subject_labels, padding=np.zeros(2))
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_object_labels,
                                                           padding=np.zeros((3, 2)))
                    yield [
                              batch_token_ids, batch_segment_ids,
                              batch_subject_labels, batch_subject_ids, batch_object_labels

                          ], None
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


def evaluate(tokenizer, data, predict):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()

    class SPO(tuple):
        """用来存三元组的类
        表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
        使得在判断两个三元组是否等价时容错性更好。
        """

        def __init__(self, spo):
            self.spox = (
                tuple(spo[0]),
                spo[1],
                tuple(spo[2]),
            )

        def __hash__(self):
            return self.spox.__hash__()

        def __eq__(self, spo):
            return self.spox == spo.spox

    for d in data:
        R = set([SPO(spo) for spo in
                 [[tokenizer.tokenize(spo_str[0][0]), spo_str[1], tokenizer.tokenize(spo_str[2][0])] for
                  spo_str
                  in predict(d['text'])]])
        T = set([SPO(spo) for spo in
                 [[tokenizer.tokenize(spo_str['s']['entity']), spo_str['p']['entity'],
                   tokenizer.tokenize(spo_str['o']['entity'])] for spo_str
                  in d['new_spo_list']]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
                             (f1, precision, recall))
        s = json.dumps(
            {
                'text': d['text'],
                'spo_list': list(T),
                'spo_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            },
            ensure_ascii=False,
            indent=4)
        f.write(s + '/n')
    pbar.close()
    f.close()
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """

    def __init__(self, model, model_path, tokenizer, predict, optimizer, valid_data):
        self.EMAer = optimizer
        self.best_val_f1 = 0.
        self.model = model
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.predict = predict
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        self.EMAer.apply_ema_weights()
        f1, precision, recall = evaluate(self.tokenizer, self.valid_data, self.predict)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(self.model_path)
        self.EMAer.reset_old_weights()
        print('f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f/n' %
              (f1, precision, recall, self.best_val_f1))
