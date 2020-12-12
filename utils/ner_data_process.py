#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/10 9:56 
# ide： PyCharm
import keras
from bert4keras.backend import K
from bert4keras.snippets import DataGenerator, sequence_padding, ViterbiDecoder
from tqdm import tqdm

from utils.logger import logger


def data_process(filename):
    D = []
    flags = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                c = c.strip()
                if len(c.split(' ')) == 1:
                    continue
                char, this_flag = c.split(' ')
                flags.append(this_flag)
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    flags = list(set(flags))
    flags = list(set([item.split('-')[1] for item in flags if len(item.split('-')) > 1]))
    return flags, D


class NerDataGenerator(DataGenerator):
    """数据生成器
    """

    def __init__(self, data, batch_size, tokenizer, label2id, maxlen):
        super().__init__(data, batch_size=batch_size)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.maxlen = maxlen

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [self.tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = self.tokenizer.encode(w)[0][1:-1]  # 只获取token_ids 不包含cls和sep
                if len(token_ids) + len(w_token_ids) < self.maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = self.label2id[l] * 2 + 1  # 防止当某个标签为0时和‘O’的标签id冲突故而使除O外的标签id>0
                        I = self.label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [self.tokenizer._token_end_id]  # ['[CLS]']+[ids] +['[SEP]']
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def evaluate(data, recognize):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def test_evaluate(data, recognize, label2id):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    xyz_dict = {label: {'X': 0, 'Y': 1e-10, 'Z': 1e-10} for label in list(label2id.keys())}
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R_p = recognize(text)
        R = set(R_p)
        T_t = [tuple(i) for i in d if i[1] != 'O']
        T = set(T_t)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        # 按标签统计
        for t in [tuple(i) for i in T_t if i[1] != 'O']:
            if t in R_p:
                R_p.remove(t)
                xyz_dict[t[1]]['X'] += 1  # 标签label预测正确TP
                xyz_dict[t[1]]['Y'] += 1  # 标签label预测正确TP
            xyz_dict[t[1]]['Z'] += 1  # 标签label真实数量TP+FN
        for p in R_p:
            xyz_dict[p[1]]['Y'] += 1  # 标签label预测伪真FP
    label_fpr = {
        label: {'f1': 2 * xyz['X'] / (xyz['Y'] + xyz['Z']), 'precision': xyz['X'] / xyz['Y'],
                'recall': xyz['X'] / xyz['Z'], 'total': xyz['X']}
        for label, xyz in xyz_dict.items()}
    for label, fpr in label_fpr.items():
        logger.info(
            '%s:  f1: %.5f, precision: %.5f, recall: %.5f   total: %d\n' %
            (label, fpr['f1'], fpr['precision'], fpr['recall'], fpr['total'])
        )
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self, model, model_path, CRF, NER, recognize, label2id, valid_data, test_data=None):
        self.best_val_f1 = 0
        self.model = model
        self.model_path = model_path
        self.CRF = CRF
        self.NER = NER
        self.recognize = recognize
        self.valid_data = valid_data
        self.test_data = test_data
        self.label2id = label2id

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(self.CRF.trans)
        self.NER.trans = trans
        # print(self.NER.trans)
        f1, precision, recall = evaluate(self.valid_data, self.recognize)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(self.model_path)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f \n' %
            (f1, precision, recall, self.best_val_f1)
        )

    def on_train_end(self, logs=None):
        if self.test_data:
            f1, precision, recall = test_evaluate(self.test_data, self.recognize, self.label2id)
            print(
                'all_test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
                (f1, precision, recall)
            )
        else:
            print('Done!')


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def __init__(self, trans, tokenizer=None, model=None, id2label=None, starts=None, ends=None):
        self.tokenizer = tokenizer
        self.model = model
        self.id2label = id2label
        super().__init__(trans, starts, ends)

    def recognize(self, text):
        tokens = self.tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        nodes = self.model.predict([[token_ids], [segment_ids]])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], self.id2label[(label - 1) // 2]])  # [[B_index],label]
                elif starting:
                    entities[-1][0].append(i)  # [[B_index,I_index,...I_index],label]
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)  # [ ('string',label),...]
                for w, l in entities]

    def batch_recognize(self, text: [], maxlen=None):
        ret = []
        batch_token_ids, batch_segment_ids, batch_token = [], [], []
        for sentence in text:
            tokens = self.tokenizer.tokenize(sentence, max_length=maxlen)
            while len(tokens) > 512:
                tokens.pop(-2)
            batch_token.append(tokens)
            token_ids, segment_ids = self.tokenizer.encode(sentence, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
        batch_token_ids = sequence_padding(batch_token_ids)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        nodes = self.model.predict([batch_token_ids, batch_segment_ids])
        for index, node in enumerate(nodes):
            pre_dict = []
            labels = self.decode(node)
            arguments, starting = [], False
            for i, label in enumerate(labels):
                if label > 0:
                    if label % 2 == 1:
                        starting = True
                        arguments.append([[i], self.id2label[str((label - 1) // 2)]])
                    elif starting:
                        arguments[-1][0].append(i)
                    else:
                        starting = False
                else:
                    starting = False
            pre_ = [
                (self.tokenizer.decode(batch_token_ids[index, w[0]:w[-1] + 1], batch_token[index][w[0]:w[-1] + 1]), l,
                 search(
                     self.tokenizer.decode(batch_token_ids[index, w[0]:w[-1] + 1], batch_token[index][w[0]:w[-1] + 1]),
                     text[index]))
                for w, l in arguments]

            ret.append(pre_)
        return ret


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1
