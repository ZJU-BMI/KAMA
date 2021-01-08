from utils import *
import os
import copy
from model.gcn import GCN


class KnowledgeGraph(object):
    """
        this is a knowledge graph

        :param data_dir: data dir of knowledge graph
        :param hidden: hidden layers of knowledge graph model
        :param train_flag: KnowledgeGraph needs training or not
        self._node: nodes of knowledge graph
        self._edge: edge of knowledge graph
        self._matrix_a: adjacent matrix of knowledge graph
        self._triple_label: triple is real or fake
        self._triple: original triple
        self._new_triple: triple after negative sampling
        self._new_triple_label: triple after negative sampling is real or fake

        """
    def __init__(self, data_dir, hidden, train_flag=False):

        self._data_dir = data_dir
        self._hidden = hidden
        # create knowledge graph
        self._node = dict()
        self._edge = dict()
        self._matrix_a = dict()
        self._matrix_d_t = dict()
        self._triple_label = []
        self._triple = []
        self._new_triple = []
        self._new_triple_label = []

        self._init()
        if train_flag:
            # 训练知识图谱
            self._entity_embedding = self._get_entity_embedding()
        else:
            if not os.path.exists(KNOWLEDGE_GRAPH_EMBEDDING_SAVE_DIR):
                raise ValueError("知识图谱节点向量文件不存在,需要重新训练")
            self._entity_embedding = pd.read_excel(KNOWLEDGE_GRAPH_EMBEDDING_SAVE_DIR)

    def _init(self):
        # load data
        whole_data = pd.read_excel(self._data_dir, sheet_name=KNOWLEDGE_GRAPH_DATA_SHEET)
        node_data = whole_data[KNOWLEDGE_GRAPH_DATA_SHEET[0]].values
        relation_data = whole_data[KNOWLEDGE_GRAPH_DATA_SHEET[1]].values

        self._triple_label = np.ones(len(relation_data))
        for index in range(len(node_data)):
            single_node = dict()
            single_node[INDEX] = index
            single_node[TYPE] = node_data[index][2]
            self._node[node_data[index][0]] = single_node

            single_relation_dict = dict()
            single_relation_dict[TARGET] = []
            single_relation_dict[TYPE] = []
            self._edge[index] = single_relation_dict

        matrix_a = np.zeros([len(node_data), len(node_data)])
        for index in range(len(relation_data)):
            single_relation = relation_data[index]
            self._triple.append([self._node[single_relation[0]][INDEX], RELATION_DICT[single_relation[1]],
                                 self._node[single_relation[2]][INDEX]])
            self._edge[self._node[single_relation[0]][INDEX]][TARGET].append(self._node[single_relation[2]][INDEX])
            self._edge[self._node[single_relation[0]][INDEX]][TYPE].append(
                single_relation[1])
            matrix_a[self._node[single_relation[0]][INDEX]][self._node[single_relation[2]][INDEX]] \
                = RELATION_DICT[single_relation[1]]

        for relation_type in RELATION_DICT:
            this_matrix_a = 1*(matrix_a == RELATION_DICT[relation_type])
            self._matrix_a[relation_type] = this_matrix_a
            self._matrix_d_t[relation_type] = self._get_matrix_degree_inverse(this_matrix_a)
        self._matrix_a[WHOLE] = 1*(matrix_a > 0)
        self._matrix_a[SELF_LOOP] = np.identity(len(self._node))

        # 负采样 正样本：负样本 = 1：5
        self.negative_sampling(5)

    @staticmethod
    def _get_matrix_degree_inverse(matrix):
        # 返回度矩阵的逆矩阵
        d = np.array(np.sum(matrix, axis=1))
        d[d == 0] = 1
        return np.linalg.inv(np.diag(d))

    def negative_sampling(self, k):
        # 负采样 正样本：负样本 = 1：k
        triple = list(self._triple)
        triple_label = list(self._triple_label)
        for source_entity in self._edge:
            source_entity_result = self._edge[source_entity]
            target_entity_list = source_entity_result[TARGET]
            new_target_entity_list = copy.deepcopy(target_entity_list)
            if len(target_entity_list) > 0:
                for target_entity_index in range(len(target_entity_list)):
                    for index in range(k):
                        temp = np.random.randint(69)
                        while temp in new_target_entity_list:
                            temp = np.random.randint(69)
                        new_target_entity_list.append(temp)
                        triple.append([source_entity,
                                       RELATION_DICT[source_entity_result[TYPE][target_entity_index]], temp])
                        triple_label.append(0)
        self._new_triple = triple
        self._new_triple_label = triple_label

    def _get_entity_embedding(self):
        gcn = GCN(n_input=len(self._node), triple_input=3, matrix_a=self._matrix_a,
                  matrix_d_t=self._matrix_d_t, feature_len=self._hidden[0], hidden=self._hidden, n_class=1)
        gcn.fit(self._new_triple, self._new_triple_label)
        return gcn.get_feature_embedding_result()

    def _get_unique_nodes_from_triple_by_source_node(self, source_nodes):
        node_index_list = []
        for node in source_nodes:
            node_index = self._node[node][INDEX]
            node_index_list.append(node_index)
            node_index_list = node_index_list + self._edge[node_index][TARGET]
        return list(set(node_index_list))

    def _get_unique_nodes_only_by_source_node(self, source_nodes):
        node_index_list = []
        for node in source_nodes:
            node_index = self._node[node][INDEX]
            node_index_list.append(node_index)
        return list(set(node_index_list))

    def get_nodes_vector_by_source_node_only(self, source_nodes):
        # 返回均值向量
        node_index_list = self._get_unique_nodes_only_by_source_node(source_nodes)
        nodes_vector_list = self._entity_embedding.iloc[node_index_list, 1:]
        nodes_vector = nodes_vector_list.mean()
        return nodes_vector
    
    def get_unique_nodes_from_triple_by_source_node(self, source_nodes):
        # 返回均值向量
        node_index_list = self._get_unique_nodes_from_triple_by_source_node(source_nodes)
        nodes_vector_list = self._entity_embedding.iloc[node_index_list, 1:]
        nodes_vector = nodes_vector_list.mean()
        return nodes_vector

    def test_matrix_a(self):
        # 如果存在从 v 到 n 的边，则节点 n 是节点 v 的邻居。
        feature_vector = [[i, -i] for i in range(len(self._node))]
        feature_vector = np.asarray(feature_vector)
        r1 = np.matmul(self._matrix_a[SUPER], feature_vector)
        r2 = np.matmul(self._matrix_a[TREAT], feature_vector)
        print(r1)
        print(r2)


def main():

    knowledge = KnowledgeGraph(KNOWLEDGE_GRAPH_SOURCE_DATA_PATH, [50, 50], train_flag=True)
    e = knowledge._entity_embedding
    print(e)
    # node = knowledge._node


if __name__ == '__main__':
    main()

