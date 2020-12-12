class TrieNode:
    """
        前缀树节点-链表格式
    """
    def __init__(self):
        self.child = {}
        # 可以加一个判断条件，但是人名提取用不到
        # self.flag = 0


class TrieTree:
    """
        前缀树构建、新增关键词、关键词词语查找等
    """
    def __init__(self):
        self.root = TrieNode()

    def add_keyword_one(self, keyword):
        """
            新增一个关键词
        :param keyword: str,构建的关键词
        :return: None
        """
        node_curr = self.root
        for word in keyword:
            if node_curr.child.get(word) is None:
                node_next = TrieNode()
                node_curr.child[word] = node_next
            node_curr = node_curr.child[word]
        # 每个关键词词后边，加入end标志位
        if node_curr.child.get('end') is None:
            node_next = TrieNode()
            node_curr.child['end'] = node_next
        node_curr = node_curr.child['end']

    def add_keyword_list(self, keywords):
        """
            新增关键词s, 格式为list
        :param keyword: list, 构建的关键词
        :return: None
        """
        for keyword in keywords:
            self.add_keyword_one(keyword)

    def extract_keyword(self, sentence):
        """
            从句子中提取关键词，取得大于2个的，例如有人名"大漠帝国"，那么"大漠帝"也取得
        :param sentence: str, 输入的句子
        :return: list, 提取到的关键词
        """
        if not sentence:
            return []
        node_curr = self.root # 关键词的第一位， 每次遍历完一个后重新初始化
        word_last = sentence[-1]
        name_list = []
        name = ''
        for word in sentence:
            if node_curr.child.get(word) is None: # 查看有无后缀
                if name: # 提取到的关键词(也可能是前面的几位)
                    if node_curr.child.get('end') is not None: # 取以end结尾的关键词， 或者len(name) > 2
                        name_list.append(name)
                    node_curr = self.root # 重新初始化
                    if self.root.child.get(word):
                        name = word
                        node_curr = node_curr.child[word]
                    else:
                        name = ''
            else: # 有缀就加到name里边
                name = name + word
                node_curr = node_curr.child[word]
                if word == word_last:  # 实体结尾的情况
                    if node_curr.child.get('end') is not None:
                        name_list.append(name)
        return name_list


def get_trie_tree(keywords):
    """
        根据list关键词，初始化trie树
    :param keywords: list, input
    :return: objext, 返回实例化的trie
    """
    trie = TrieTree()
    trie.add_keyword_list(keywords)
    return trie
