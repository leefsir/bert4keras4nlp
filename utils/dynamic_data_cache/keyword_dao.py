import pandas as pd

from config.configs import ENTITY_DICT
from utils.dynamic_data_cache.trie_tree import get_trie_tree


class KeywordInitial:
    def __init__(self):
        # 从数据库获取所有关键词，并按 '，' 分割。返回（KEY, VALUE, TYPE, ID）
        self.get_entity()
        self.get_entity_tree()

    def get_entity_tree(self):
        self.entity_trees = {}
        for entity_type, entity_lists in self.entitys.items():
            self.entity_trees[entity_type] = get_trie_tree(entity_lists)

    def delete(self, keyword_tuple: [(), ]):

        for value, entity_type in keyword_tuple:
            if self.entitys.get(entity_type) and value in self.entitys.get(entity_type):
                self.entitys.get(entity_type).remove(value)
        self.get_entity_tree()

    def add(self, keyword_tuple: [(), ]):
        '''
        :param keyword_tuple: [(value, entity_type),]
        :return:
        '''
        for value, entity_type in keyword_tuple:
            if self.entitys.get(entity_type) and value not in self.entitys.get(entity_type):
                self.entitys.get(entity_type).append(value)
            elif not self.entitys.get(entity_type):
                self.entitys[entity_type] = [value]
        self.get_entity_tree()

    def update_all(self):
        self.get_entity()
        self.get_entity_tree()

    def get_entity(self, entity_path=ENTITY_DICT):
        entity_df = pd.read_csv(entity_path)
        if list(entity_df) != ['value', 'entity_type']:
            raise Exception("Incorrect format! The column name is ['value', 'entity_type'] not {}".format(list(entity_df)))
        entity_df = entity_df.drop_duplicates(subset='value', keep='first')
        self.entitys = {}
        for value, entity_type in entity_df.values.tolist():
            if self.entitys.get(entity_type):
                self.entitys.get(entity_type).append(value)
            else:
                self.entitys[entity_type] = [value]


# 项目启动就加载所有关键字
keywordinital = KeywordInitial()
