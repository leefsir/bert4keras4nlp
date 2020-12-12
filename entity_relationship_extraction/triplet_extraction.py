# _*_coding:utf-8_*_
# author leewfeng
# 2020/12/12 20:04
import os

from basis_framework.basis_graph import BasisGraph
from utils.common_tools import load_json, save_json
from utils.triplet_data_process import Data_Generator, data_process, Evaluator

rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from bert4keras.backend import K, batch_gather
from bert4keras.layers import LayerNormalization
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
import numpy as np

model_root_path = rootPath + '/model/'
corpus_root_path = rootPath + '/corpus/'


class ReextractBertHandler(BasisGraph):
    def __init__(self, params={}, Train=False):
        if not params.get('model_code'):
            params['model_code'] = 'triplet_extraction'
        super().__init__(params, Train)

    def load_params(self):
        load_params = load_json(self.params_path)
        self.max_len = load_params.get('max_len')
        self.num_classes = load_params.get('num_classes')
        self.p2s_dict = load_params.get('p2s_dict')
        self.i2p_dict = load_params.get('i2p_dict')
        self.p2o_dict = load_params.get('p2o_dict')

    def save_params(self):
        self.params['num_classes'] = self.num_classes
        self.params['p2s_dict'] = self.p2s_dict
        self.params['i2p_dict'] = self.i2p_dict
        self.params['p2o_dict'] = self.p2o_dict
        self.params['max_len'] = self.max_len
        save_json(jsons=self.params, json_path=self.params_path)

    def data_process(self):
        train_data, self.valid_data, self.p2s_dict, self.p2o_dict, self.i2p_dict, self.p2i_dict = data_process(
            self.train_data_path, self.valid_data_path, self.max_len, self.params_path)
        self.num_classes = len(self.i2p_dict)
        self.train_generator = Data_Generator(train_data, self.batch_size, self.tokenizer, self.p2i_dict,
                                              self.max_len)

    def extrac_subject(self, inputs):
        """根据subject_ids从output中取出subject的向量表征
        """
        output, subject_ids = inputs
        subject_ids = K.cast(subject_ids, 'int32')
        start = batch_gather(output, subject_ids[:, :1])
        end = batch_gather(output, subject_ids[:, 1:])
        subject = K.concatenate([start, end], 2)
        return subject[:, 0]

    def build_model(self):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
        if self.memory_fraction:
            config.gpu_options.per_process_gpu_memory_fraction = self.memory_fraction
            config.gpu_options.allow_growth = False
        else:
            config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        # 补充输入
        subject_labels = Input(shape=(None, 2), name='Subject-Labels')
        subject_ids = Input(shape=(2,), name='Subject-Ids')
        object_labels = Input(shape=(None, self.num_classes, 2), name='Object-Labels')
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=self.bert_config_path,
            checkpoint_path=self.bert_checkpoint_path,
            return_keras_model=False,
        )
        # 预测subject
        output = Dense(units=2,
                       activation='sigmoid',
                       kernel_initializer=bert.initializer)(bert.model.output)
        subject_preds = Lambda(lambda x: x ** 2)(output)
        self.subject_model = Model(bert.model.inputs, subject_preds)
        # 传入subject，预测object
        # 通过Conditional Layer Normalization将subject融入到object的预测中
        output = bert.model.layers[-2].get_output_at(-1)
        subject = Lambda(self.extrac_subject)([output, subject_ids])
        output = LayerNormalization(conditional=True)([output, subject])
        output = Dense(units=self.num_classes * 2,
                       activation='sigmoid',
                       kernel_initializer=bert.initializer)(output)
        output = Lambda(lambda x: x ** 4)(output)
        object_preds = Reshape((-1, self.num_classes, 2))(output)
        self.object_model = Model(bert.model.inputs + [subject_ids], object_preds)
        # 训练模型
        self.model = Model(bert.model.inputs + [subject_labels, subject_ids, object_labels],
                           [subject_preds, object_preds])

        mask = bert.model.get_layer('Embedding-Token').output_mask
        mask = K.cast(mask, K.floatx())
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        self.model.add_loss(subject_loss + object_loss)
        AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
        self.optimizer = AdamEMA(lr=1e-4)

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer)

    def predict(self, text):
        """
        抽取输入text所包含的三元组
        text：str(<离开>是由张宇谱曲，演唱)
        """
        tokens = self.tokenizer.tokenize(text, max_length=self.max_len)
        token_ids, segment_ids = self.tokenizer.encode(text, max_length=self.max_len)
        # 抽取subject
        subject_preds = self.subject_model.predict([[token_ids], [segment_ids]])
        start = np.where(subject_preds[0, :, 0] > 0.6)[0]
        end = np.where(subject_preds[0, :, 1] > 0.5)[0]
        subjects = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))
        if subjects:
            spoes = []
            token_ids = np.repeat([token_ids], len(subjects), 0)
            segment_ids = np.repeat([segment_ids], len(subjects), 0)
            subjects = np.array(subjects)
            # 传入subject，抽取object和predicate
            object_preds = self.object_model.predict([token_ids, segment_ids, subjects])
            for subject, object_pred in zip(subjects, object_preds):
                start = np.where(object_pred[:, :, 0] > 0.6)
                end = np.where(object_pred[:, :, 1] > 0.5)
                for _start, predicate1 in zip(*start):
                    for _end, predicate2 in zip(*end):
                        if _start <= _end and predicate1 == predicate2:
                            spoes.append((subject, predicate1, (_start, _end)))
                            break
            return [
                (
                    [self.tokenizer.decode(token_ids[0, s[0]:s[1] + 1], tokens[s[0]:s[1] + 1]),
                     self.p2s_dict[self.i2p_dict[p]]],
                    self.i2p_dict[p],
                    [self.tokenizer.decode(token_ids[0, o[0]:o[1] + 1], tokens[o[0]:o[1] + 1]),
                     self.p2o_dict[self.i2p_dict[p]]],
                    (s[0], s[1] + 1),
                    (o[0], o[1] + 1)
                ) for s, p, o in spoes
            ]
        else:
            return []

    def train(self):
        evaluator = Evaluator(self.model, self.model_path, self.tokenizer, self.predict, self.optimizer,
                              self.valid_data)

        self.model.fit_generator(self.train_generator.forfit(),
                                 steps_per_epoch=len(self.train_generator),
                                 epochs=self.epoch,
                                 callbacks=[evaluator])


if __name__ == '__main__':
    params = {
        "max_len": 128,
        "batch_size": 32,
        "epoch": 1,
        "train_data_path": rootPath + "/data/train_data.json",
        "dev_data_path": rootPath + "/data/valid_data.json",
    }

    model = ReextractBertHandler(params, Train=True)

    model.train()
    text = "马志舟，1907年出生，陕西三原人，汉族，中国共产党，任红四团第一连连长，1933年逝世"
    print(model.predict(text))
