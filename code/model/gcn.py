from utils import *
import tensorflow as tf

class GCN(object):
    def __init__(self, n_input, triple_input, matrix_a, matrix_d_t, feature_len, hidden, n_class,
                 transfer=tf.nn.sigmoid, optimizer=tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer),
                 epochs=1000, name='gcn', sess=None):
        """
        simple GCN

        :param n_input: num of input node
        :param triple_input:
        :param matrix_a: adjacency matrix
        :param matrix_d_t: degree inverse matrix
        :param feature_len: init feature node len
        :param hidden: list of hidden layer node num
        :param n_class: num of output node
        :param optimizer: optimization algorithm
        :param epochs: training epoch
        :param name: model name
        :param sess: tf.sess
        :param
        """
        self._n_input = n_input
        self._triple_input = triple_input
        self._hidden = hidden
        self._matrix_a = dict()
        self._matrix_d_t = dict()
        self._n_class = n_class
        self._epochs = epochs
        self._name = name
        self._feature_len = feature_len

        self._feature_embedding_result = None
        with tf.variable_scope(self._name):
            self._transfer_func = transfer
            self._optimizer = optimizer(weight_decay=0.001, learning_rate=0.001)
            self._n_epoch = 0
            self._sess = sess if sess is not None else tf.Session()
            # transform np to sensor
            self._matrix_transform(matrix_a, matrix_d_t)
            self._graph_definition()

    def _matrix_transform(self, matrix_a, matrix_d_t):
        # transform np to sensor
        for relation_type in RELATION_LIST:
            self._matrix_a[relation_type] = tf.convert_to_tensor(matrix_a[relation_type], dtype=tf.float32)
            if relation_type != SELF_LOOP:
                self._matrix_d_t[relation_type] = tf.convert_to_tensor(matrix_d_t[relation_type], dtype=tf.float32)

    def _graph_definition(self):
        self._triple = tf.placeholder(tf.int32, [None, self._triple_input], name="triple")
        self._triple_label = tf.placeholder(tf.float32, [None, self._n_class], name="triple_label")
        self._score = []

        self._feature_vector = tf.Variable(xavier_init(self._n_input, self._hidden[0]), dtype=tf.float32,
                                           name="feature_vector")
        # 1 到 k 层的隐藏层 每一层对应五种关系有五种W权重
        # 知识图谱中一共有四种关系。第五种是SELF-LOOP，特别用于计算
        self._W = dict()
        for hidden_num in range(len(self._hidden)):
            if hidden_num == 0:
                w = dict()
                for relation_type in RELATION_LIST:
                    w[relation_type] = tf.Variable(xavier_init(self._feature_len, self._hidden[hidden_num]), dtype=tf.float32,
                                           name='hidden{}_weight{}'.format(hidden_num, relation_type))

                self._W[hidden_num] = w
            else:
                w = dict()
                for relation_type in RELATION_LIST:
                    w[relation_type] = tf.Variable(xavier_init(self._hidden[hidden_num-1], self._hidden[hidden_num]),
                                                   dtype=tf.float32, name='hidden{}_weight{}'.format(hidden_num, relation_type))
                self._W[hidden_num] = w

        # 通过关系矩阵进行卷积过程
        self._hidden_output = []
        for hidden_num in range(len(self._hidden)):
            if hidden_num == 0:
                feature_matrix = self._feature_vector
                feature_matrix_super = tf.matmul(self._matrix_a[SUPER], feature_matrix) @  self._W[hidden_num][SUPER]
                feature_matrix_treat = tf.matmul(self._matrix_a[TREAT], feature_matrix) @  self._W[hidden_num][TREAT]
                feature_matrix_hint = tf.matmul(self._matrix_a[HINT], feature_matrix) @  self._W[hidden_num][HINT]
                feature_matrix_cause = tf.matmul(self._matrix_a[CAUSE], feature_matrix) @  self._W[hidden_num][CAUSE]
                feature_matrix_self_loop = tf.matmul(self._matrix_a[SELF_LOOP], feature_matrix) @  self._W[hidden_num][
                    SELF_LOOP]
                feature_matrix_out = self._matrix_d_t[SUPER] @ feature_matrix_super + self._matrix_d_t[
                    TREAT] @ feature_matrix_treat + self._matrix_d_t[HINT] @ feature_matrix_hint + self._matrix_d_t[
                                         CAUSE] @ feature_matrix_cause + feature_matrix_self_loop

                feature_matrix_out = tf.nn.relu(feature_matrix_out)
                self._hidden_output.append(feature_matrix_out)
            else:
                feature_matrix = self._hidden_output[hidden_num-1]
                feature_matrix_super = tf.matmul(self._matrix_a[SUPER], feature_matrix) @ self._W[hidden_num][SUPER]
                feature_matrix_treat = tf.matmul(self._matrix_a[TREAT], feature_matrix) @ self._W[hidden_num][TREAT]
                feature_matrix_hint = tf.matmul(self._matrix_a[HINT], feature_matrix) @ self._W[hidden_num][HINT]

                feature_matrix_cause = tf.matmul(self._matrix_a[CAUSE], feature_matrix) @ self._W[hidden_num][CAUSE]
                feature_matrix_self_loop = tf.matmul(self._matrix_a[SELF_LOOP], feature_matrix) @ self._W[hidden_num][
                    SELF_LOOP]
                feature_matrix_out = self._matrix_d_t[SUPER] @ feature_matrix_super + self._matrix_d_t[
                    TREAT] @ feature_matrix_treat + self._matrix_d_t[HINT] @ feature_matrix_hint + self._matrix_d_t[
                                         CAUSE] @ feature_matrix_cause + feature_matrix_self_loop
                feature_matrix_out = tf.nn.relu(feature_matrix_out)
                self._hidden_output.append(feature_matrix_out)
        self._feature_matrix_final_output = self._hidden_output[-1]
        # DistMult factorization
        self._relation = dict()
        diagonal_mask = tf.eye(self._hidden[-1])
        for relation_type in RELATION_DICT:
            realtion_matrix = tf.Variable(xavier_init(self._hidden[-1], self._hidden[-1]), dtype=tf.float32,
                                           name='relation_{}'.format(relation_type))
            # 确保是diagonal
            self._relation[relation_type] = tf.multiply(diagonal_mask, realtion_matrix)

        self._score, self._label = self.score()
        self._loss_links_predict = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self._label, logits=self._score))
        self._generator_train_op = self._optimizer.minimize(self._loss_links_predict)

    def score(self):
        # 评分
        # 1.单个三元组逐次计算
        # 2.五种关系 矩阵计算求和 取对角线(需要对应的mask)

        # tf.nn.embedding_lookup(params, ids, max_norm=None, name=None)
        # 根据ID返回实体的embedding
        score = []
        label = []
        source_entity = tf.nn.embedding_lookup(self._feature_matrix_final_output, self._triple[:, 0], name=None)
        target_entity = tf.nn.embedding_lookup(self._feature_matrix_final_output, self._triple[:, 2], name=None)
        index = 0
        ralation_index = dict()
        for relation_type in RELATION_DICT:
            # 获取五种关系的index
            ralation_index[relation_type] = tf.where(tf.equal(self._triple[:, 1], RELATION_DICT[relation_type]))
            new_source_entity = tf.gather(source_entity, ralation_index[relation_type])
            new_target_entity = tf.gather(target_entity, ralation_index[relation_type])
            new_source_entity = tf.squeeze(new_source_entity)
            new_target_entity = tf.squeeze(new_target_entity)
            score_matrix = new_source_entity @ self._relation[relation_type] @ tf.matrix_transpose(new_target_entity)
            if relation_type == SUPER:
                # 获得矩阵对角线的值(sum(es[i]*r[i]*et[i]))
                score = tf.diag_part(score_matrix)
                # gather 相应的 source_entity_label
                label = tf.gather(self._triple_label, ralation_index[relation_type])
            else:
                score = tf.concat([score, tf.diag_part(score_matrix)], 0)
                label = tf.concat([label, tf.gather(self._triple_label, ralation_index[relation_type])], 0)

        label = tf.squeeze(label)
        score = tf.squeeze(score)
        # for index in range(source_entity.shape[0]):
        #     single_score = source_entity[index].T @ self._relation[self._triple[index, 2]] @ target_entity[index]
        #     score.append(tf.nn.softmax(single_score))
        return score, label

    def get_feature_embedding_result(self):
        return self._feature_embedding_result

    def save_feature_embedding_result(self, feature_embedding):
        self._feature_embedding_result = pd.DataFrame(feature_embedding, index=None)
        self._feature_embedding_result.to_excel(KNOWLEDGE_GRAPH_EMBEDDING_SAVE_DIR)

    def fit(self, triple, triple_label):
        triple_label = np.asarray(triple_label).reshape([-1, 1])
        self._sess.run(tf.global_variables_initializer())
        while self._n_epoch < self._epochs:
            loss_links_predict, _ = self._sess.run((self._loss_links_predict, self._generator_train_op),
                                                feed_dict={self._triple: triple,
                                                           self._triple_label: triple_label})
            score, label = self._sess.run((self._score, self._label),
                                          feed_dict={self._triple: triple, self._triple_label: triple_label})
            label = np.reshape(label, [-1, ])
            target_test_auc = roc_auc_score(label, score)
            print(loss_links_predict, target_test_auc)
            self._n_epoch = self._n_epoch + 1

        feature_embedding = self._sess.run(self._feature_matrix_final_output,
                                                   feed_dict={self._triple: triple,
                                                              self._triple_label: triple_label})
        self.save_feature_embedding_result(feature_embedding)






