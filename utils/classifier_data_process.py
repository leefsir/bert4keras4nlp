#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 17:22 
# ide： PyCharm

import keras
from bert4keras.snippets import DataGenerator, sequence_padding
from tqdm import tqdm


# DataGenerator只是一种为了节约内存的数据方式
class Data_Generator(DataGenerator):
    def __init__(self, data, l2i, tokenizer, batch_size, maxlen=128):
        super().__init__(data, batch_size=batch_size)
        self.l2i = l2i
        self.maxlen = maxlen
        self.tokenizer = tokenizer

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (label, text) in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(text, max_length=self.maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([self.l2i.get(str(label))])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def evaluate(data, predict):
    total, right = 0., 0.
    for x_true, y_true in tqdm(data):
        # for x_true, y_true in data:
        y_pred = predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self, model, model_path, valid_generator, test_generator):
        self.best_val_acc = 0.
        self.model = model
        self.model_path = model_path
        self.valid_generator = valid_generator
        self.test_generator = test_generator

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(self.valid_generator, self.model.predict)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.model_path)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

    def on_train_end(self, logs=None):
        test_acc = evaluate(self.test_generator, self.model.predict)
        print(
            u'best_val_acc: %.5f, test_acc: %.5f\n' %
            (self.best_val_acc, test_acc)
        )
