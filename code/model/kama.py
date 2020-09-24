from utils import *


class KAMA(object):
    def __init__(self, n_input, feature_encoder_hidden, feature_embedding_input, n_class,
                 transfer=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer(),
                 epochs=300, name='kama', sess=None):
        """
        a Knowledge-Aware Multi-center clinical dataset Adaptation model

        :param n_input: num of input node
        :param feature_encoder_hidden: list of hidden feature encoder layer node num
        :param n_class: num of output node
        :param optimizer: optimization algorithm
        :param epochs: training epoch
        :param name: model name
        :param sess: tf.sess
        :param
        """
        self._n_input = n_input
        self._feature_encoder_hidden = feature_encoder_hidden
        self._feature_embedding_input = feature_embedding_input
        self._n_class = n_class
        self._epochs = epochs
        self._name = name

        with tf.variable_scope(self._name):
            self._transfer_func = transfer
            self._optimizer = optimizer
            # output_n_epoch: now training epoch
            self._output_n_epoch = 0
            self._sess = sess if sess is not None else tf.Session()
            self._graph_definition()

    def _u_init(self):
        with tf.variable_scope("u"):
            u = dict()
            u['up'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="up")
            u['uk'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="uk")
            return u

    def _attention_definition(self, zp, zk):
        source_correlation_matrix = tf.matmul(tf.transpose(zp), tf.nn.tanh(zk))
        self._ap = tf.reshape(tf.nn.tanh(tf.reduce_mean(source_correlation_matrix, axis=1)), [1, -1])
        self._ak = tf.reshape(tf.nn.tanh(tf.reduce_mean(source_correlation_matrix, axis=0)), [1, -1])
        ip = tf.ones_like(zp[:, 0:1])
        ik = tf.ones_like(zk[:, 0:1])
        self._bp = tf.nn.tanh((zp + (ip @ self._ap) * zk) @ self._u['up'])
        self._bk = tf.nn.tanh((zk + (ik @ self._ak) * zp) @ self._u['uk'])
        pai_p = self._bp * zp
        pai_k = self._bk * zk
        return tf.concat([pai_p, pai_k], axis=1)

    def _weights_init(self):
        with tf.variable_scope("dae_weights"):
            weights = dict()
            weights['w1'] = tf.Variable(xavier_init(self._feature_embedding_input,
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="w1")
            weights['b1'] = tf.Variable(tf.zeros(self._feature_encoder_hidden[-1]), name="b1")
            weights['w2'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_embedding_input), dtype=tf.float32, name="w2")
            weights['b2'] = tf.Variable(tf.zeros(self._feature_embedding_input), name="b2")
            return weights

    def _graph_definition(self):
        self._source_input = tf.placeholder(tf.float32, [None, self._n_input], name="source_input")
        self._target_input = tf.placeholder(tf.float32, [None, self._n_input], name="target_input")
        self._source_label = tf.placeholder(tf.float32, [None, self._n_class], name="source_label")
        self._target_label = tf.placeholder(tf.float32, [None, self._n_class], name="target_label")

        self._source_xk = tf.placeholder(tf.float32, [None, self._feature_embedding_input], name="source_xk")
        self._target_xk = tf.placeholder(tf.float32, [None, self._feature_embedding_input], name="target_xk")
        self._keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("generator"):
            # knowledge feature extraction
            self._dae_weights = self._weights_init()
            # knowledge feature encoder
            self._source_zk = self._source_xk @ self._dae_weights['w1'] + self._dae_weights['b1']
            self._target_zk = self._target_xk @ self._dae_weights['w1'] + self._dae_weights['b1']
            # knowledge feature decoder
            self._source_reconstruction = self._source_zk @ self._dae_weights['w2'] + self._dae_weights['b2']
            self._target_reconstruction = self._target_zk @ self._dae_weights['w2'] + self._dae_weights['b2']

            self._loss_of_reconstruction = tf.reduce_mean(
                tf.losses.mean_squared_error(self._source_xk, self._source_reconstruction)) + tf.reduce_mean(
                tf.losses.mean_squared_error(self._target_xk, self._target_reconstruction))
            # tensor after feature encoder
            with tf.variable_scope('feature_encoder', reuse=tf.AUTO_REUSE):
                self._source_feature_encoder_output = self._init_feature_encoder(self._source_input)
                self._target_feature_encoder_output = self._init_feature_encoder(self._target_input)
            # attention
            # correlation_matrix M
            with tf.variable_scope('knowledge-aware_patient_representation', reuse=tf.AUTO_REUSE):
                self._u = self._u_init()
                self._source_patient_representation = self._attention_definition(self._source_feature_encoder_output,
                                                                                 self._source_zk)
                self._target_patient_representation = self._attention_definition(self._target_feature_encoder_output,
                                                                                 self._target_zk)
                # prediction of two data set
        with tf.variable_scope("prognosis", reuse=tf.AUTO_REUSE):
                self._source_prognosis_output, self._source_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._source_patient_representation, name="prognosis")
                self._target_prognosis_output, self._target_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._target_patient_representation, name="prognosis")
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            self._source_discriminator_output, self._source_discriminator_predict = self._hidden_layer_of_discriminator(
                self._source_patient_representation, name="discriminator")
            self._target_discriminator_output, self._target_discriminator_predict = self._hidden_layer_of_discriminator(
                self._target_patient_representation, name="discriminator")

        # discriminator loss
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._target_discriminator_output,
                                                    labels=tf.zeros_like(self._target_discriminator_output)))
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._source_discriminator_output,
                                                    labels=tf.ones_like(self._source_discriminator_output)))
        self._loss_of_discriminator = tf.add(fake_loss, real_loss)
        # generator loss
        self._gen_loss = self._loss_of_discriminator
        self._loss_classification = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self._source_label, logits=self._source_prognosis_output))
        self._loss_of_whole_generator = self._loss_classification - 0.5 * self._gen_loss + self._loss_of_reconstruction
        # self._loss_of_whole_generator = self._loss_of_reconstruction
        self._loss_of_whole_discriminator = self._loss_of_discriminator

        train_vars = tf.trainable_variables()
        # 分类器变量
        clf_vars = [var for var in train_vars if var.name.startswith('kama/prognosis')]
        print(clf_vars)
        # 生成器变量
        gen_vars = [var for var in train_vars if var.name.startswith('kama/generator')]
        print(gen_vars)
        # 判别器变量
        dis_vars = [var for var in train_vars if var.name.startswith('kama/discriminator')]
        print(dis_vars)
        self._generator_train_op = self._optimizer.minimize(self._loss_of_whole_generator, var_list=gen_vars)
        self._discriminator_train_op = self._optimizer.minimize(self._loss_of_discriminator, var_list=dis_vars)
        self._clf_train_op = self._optimizer.minimize(self._loss_classification, var_list=clf_vars)

    def _init_feature_encoder(self, input_data):
        hidden_outputs = []
        for i in range(len(self._feature_encoder_hidden)):
            if i == 0:
                hidden_output = tf.layers.dense(input_data, self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.tanh)

            else:
                hidden_output = tf.layers.dense(hidden_outputs[i-1], self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.tanh)

            hidden_outputs.append(hidden_output)
        embedding_output = tf.layers.dropout(inputs=hidden_outputs[-1], rate=1-self._keep_prob)
        return embedding_output

    def fit(self, source_train_set, source_test_set, target_train_set, target_test_set, batch_size, keep_prob):
        train_data = []
        self._sess.run(tf.global_variables_initializer())
        for c in tf.trainable_variables(self._name):
            print(c.name)
        while source_train_set.epoch_completed < self._epochs:
            source_train_input, source_train_task, source_train_embedding = source_train_set.next_batch(batch_size)
            source_train_task = np.reshape(source_train_task, [-1, 1])
            target_train_input, target_train_task, target_train_embedding = target_train_set.next_batch(batch_size)
            # 损失函数
            loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss, loss_of_reconstruction\
                = self._sess.run(
                (self._loss_of_whole_generator, self._loss_of_discriminator,
                 self._loss_classification, self._gen_loss, self._loss_of_reconstruction),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob,
                           self._source_xk: source_train_embedding, self._target_xk: target_train_embedding})
            # 最小化训练
            self._sess.run(
                (self._generator_train_op, self._clf_train_op, self._discriminator_train_op),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob,
                           self._source_xk: source_train_embedding, self._target_xk: target_train_embedding})
            if self._output_n_epoch < source_train_set.epoch_completed:
                source_test_prediction = self.predict(source_test_set.x, source_test_set.xk)
                target_test_prediction = self.predict(target_test_set.x, target_test_set.xk)

                target_test_auc, target_test_precision, target_test_recall, target_test_f_score, target_test_accuracy, target_test_fpr, target_test_tpr, target_test_thresholds,target_test_prediction_label \
                    = calculate_score_and_get_roc(target_test_set.y, target_test_prediction)
                source_test_auc, source_test_precision, source_test_recall, source_test_f_score, source_test_accuracy, source_test_fpr, source_test_tpr, source_test_thresholds,source_test_prediction_label \
                    = calculate_score_and_get_roc(source_test_set.y, source_test_prediction)
                self._output_n_epoch = self._output_n_epoch + 1
                train_data.append([source_test_auc, source_test_precision, source_test_recall, source_test_f_score,
                                   source_test_accuracy, target_test_auc, target_test_precision, target_test_recall,
                                   target_test_f_score, target_test_accuracy, source_test_auc + target_test_auc])
                print("epoch:{}\tsource_test_auc:{}\ttarget_test_auc:{}\tloss_of_whole_generator:{}\t"
                      "loss_of_discriminator:{}\tloss_classification:{}\tgen_loss:{}\tloss_of_reconstruction:{}\t".
                      format(source_train_set.epoch_completed, source_test_auc, target_test_auc,
                             loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss,
                             loss_of_reconstruction))
        train_data = np.array(train_data)
        max_i = np.where(train_data[:, -1] == train_data[:, -1].max())
        print(np.squeeze(train_data[max_i, :]))
        return np.squeeze(train_data[max_i, :])

    def predict(self, x, xk):
        return self._sess.run(
                    self._source_prognosis_predict,
                    feed_dict={self._source_input: x, self._keep_prob: 1,
                               self._source_xk: xk})

    @staticmethod
    def _hidden_layer_of_prognosis(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 10, activation=tf.nn.tanh, name="first_hidden_output")
        prognosis_output = tf.layers.dense(first_hidden_output, 1, name=name)
        prognosis_predict = tf.nn.sigmoid(prognosis_output)
        return prognosis_output, prognosis_predict

    @staticmethod
    def _hidden_layer_of_discriminator(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 10, activation=tf.nn.tanh, name="first_hidden_output")
        discriminator_output = tf.layers.dense(first_hidden_output, 1, name=name)
        discriminator_predict = tf.nn.sigmoid(discriminator_output)
        return discriminator_output, discriminator_predict


class KAMA_V1(object):
    def __init__(self, n_input, feature_encoder_hidden, feature_embedding_input, n_class,
                 transfer=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer(),
                 epochs=300, name='kama_v1', sess=None):
        """
        a Knowledge-Aware Multi-center clinical dataset Adaptation model Variant 1
        Variant 1 uses separate patient feature encoders

        :param n_input: num of input node
        :param feature_encoder_hidden: list of hidden feature encoder layer node num
        :param n_class: num of output node
        :param optimizer: optimization algorithm
        :param epochs: training epoch
        :param name: model name
        :param sess: tf.sess
        :param
        """
        self._n_input = n_input
        self._feature_encoder_hidden = feature_encoder_hidden
        self._feature_embedding_input = feature_embedding_input
        self._n_class = n_class
        self._epochs = epochs
        self._name = name

        with tf.variable_scope(self._name):
            self._transfer_func = transfer
            self._optimizer = optimizer
            # output_n_epoch: now training epoch
            self._output_n_epoch = 0
            self._sess = sess if sess is not None else tf.Session()
            self._graph_definition()

    def _u_init(self):
        with tf.variable_scope("u"):
            u = dict()
            u['up'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="up")
            u['uk'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="uk")
            return u

    def _attention_definition(self, zp, zk):
        source_correlation_matrix = tf.matmul(tf.transpose(zp), tf.nn.tanh(zk))
        self._ap = tf.reshape(tf.nn.tanh(tf.reduce_mean(source_correlation_matrix, axis=1)), [1, -1])
        self._ak = tf.reshape(tf.nn.tanh(tf.reduce_mean(source_correlation_matrix, axis=0)), [1, -1])
        ip = tf.ones_like(zp[:, 0:1])
        ik = tf.ones_like(zk[:, 0:1])
        self._bp = tf.nn.tanh((zp + (ip @ self._ap) * zk) @ self._u['up'])
        self._bk = tf.nn.tanh((zk + (ik @ self._ak) * zp) @ self._u['uk'])
        pai_p = self._bp * zp
        pai_k = self._bk * zk
        return tf.concat([pai_p, pai_k], axis=1)

    def _weights_init(self):
        with tf.variable_scope("dae_weights"):
            weights = dict()
            weights['ws1'] = tf.Variable(xavier_init(self._feature_embedding_input,
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="ws1")
            weights['bs1'] = tf.Variable(tf.zeros(self._feature_encoder_hidden[-1]), name="bs1")
            weights['ws2'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_embedding_input), dtype=tf.float32, name="ws2")

            weights['bs2'] = tf.Variable(tf.zeros(self._feature_embedding_input), name="bs2")
            return weights



    def _graph_definition(self):
        self._source_input = tf.placeholder(tf.float32, [None, self._n_input], name="source_input")
        self._target_input = tf.placeholder(tf.float32, [None, self._n_input], name="target_input")
        self._source_label = tf.placeholder(tf.float32, [None, self._n_class], name="source_label")
        self._target_label = tf.placeholder(tf.float32, [None, self._n_class], name="target_label")

        self._source_xk = tf.placeholder(tf.float32, [None, self._feature_embedding_input], name="source_xk")
        self._target_xk = tf.placeholder(tf.float32, [None, self._feature_embedding_input], name="target_xk")
        self._keep_prob = tf.placeholder(tf.float32)


        with tf.variable_scope("generator"):
            # knowledge feature extraction
            self._dae_weights = self._weights_init()
            # knowledge feature encoder
            self._source_zk = self._source_xk @ self._dae_weights['ws1'] + self._dae_weights['bs1']
            self._target_zk = self._target_xk @ self._dae_weights['ws1'] + self._dae_weights['bs1']
            # knowledge feature decoder
            self._source_reconstruction = self._source_zk @ self._dae_weights['ws2'] + self._dae_weights['bs2']
            self._target_reconstruction = self._target_zk @ self._dae_weights['ws2'] + self._dae_weights['bs2']

            self._loss_of_reconstruction = tf.reduce_mean(
                tf.losses.mean_squared_error(self._source_xk, self._source_reconstruction)) + tf.reduce_mean(
                tf.losses.mean_squared_error(self._target_xk, self._target_reconstruction))
            # tensor after feature encoder
            with tf.variable_scope('source_feature_encoder', reuse=tf.AUTO_REUSE):
                self._source_feature_encoder_output = self._init_feature_encoder(self._source_input)

            with tf.variable_scope('target_feature_encoder', reuse=tf.AUTO_REUSE):
                self._target_feature_encoder_output = self._init_feature_encoder(self._target_input)
            # attention
            # correlation_matrix M
            with tf.variable_scope('knowledge-aware_patient_representation', reuse=tf.AUTO_REUSE):
                self._u = self._u_init()
                self._source_patient_representation = self._attention_definition(self._source_feature_encoder_output,
                                                                                 self._source_zk)
                self._target_patient_representation = self._attention_definition(self._target_feature_encoder_output,
                                                                                 self._target_zk)
                # prediction of two data set
        with tf.variable_scope("prognosis", reuse=tf.AUTO_REUSE):
                self._source_prognosis_output, self._source_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._source_patient_representation, name="prognosis")
                self._target_prognosis_output, self._target_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._target_patient_representation, name="prognosis")
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            self._source_discriminator_output, self._source_discriminator_predict = self._hidden_layer_of_discriminator(
                self._source_patient_representation, name="discriminator")
            self._target_discriminator_output, self._target_discriminator_predict = self._hidden_layer_of_discriminator(
                self._target_patient_representation, name="discriminator")

        # discriminator loss
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._target_discriminator_output,
                                                    labels=tf.zeros_like(self._target_discriminator_output)))
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._source_discriminator_output,
                                                    labels=tf.ones_like(self._source_discriminator_output)))
        self._loss_of_discriminator = tf.add(fake_loss, real_loss)
        self._gen_loss = self._loss_of_discriminator
        self._loss_classification = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self._source_label, logits=self._source_prognosis_output))
        self._loss_of_whole_generator = self._loss_classification - 0.5 * self._gen_loss + self._loss_of_reconstruction
        self._loss_of_whole_discriminator = self._loss_of_discriminator

        train_vars = tf.trainable_variables()
        # 分类器变量
        clf_vars = [var for var in train_vars if var.name.startswith('kama/prognosis')]
        print(clf_vars)
        # 生成器变量
        gen_vars = [var for var in train_vars if var.name.startswith('kama/generator')]
        print(gen_vars)
        # 判别器变量
        dis_vars = [var for var in train_vars if var.name.startswith('kama/discriminator')]
        print(dis_vars)
        # self._generator_train_op = self._optimizer.minimize(self._loss_of_whole_generator)
        self._generator_train_op = self._optimizer.minimize(self._loss_of_whole_generator, var_list=gen_vars)
        self._discriminator_train_op = self._optimizer.minimize(self._loss_of_discriminator, var_list=dis_vars)
        self._clf_train_op = self._optimizer.minimize(self._loss_classification, var_list=clf_vars)

    def _init_feature_encoder(self, input_data):
        hidden_outputs = []
        for i in range(len(self._feature_encoder_hidden)):
            if i == 0:
                hidden_output = tf.layers.dense(input_data, self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.tanh)

            else:
                hidden_output = tf.layers.dense(hidden_outputs[i-1], self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.tanh)

            hidden_outputs.append(hidden_output)
        embedding_output = tf.layers.dropout(inputs=hidden_outputs[-1], rate=1-self._keep_prob)
        return embedding_output

    def fit(self, source_train_set, source_test_set, target_train_set, target_test_set, batch_size, keep_prob):
        train_data = []
        self._sess.run(tf.global_variables_initializer())
        for c in tf.trainable_variables(self._name):
            print(c.name)

        while source_train_set.epoch_completed < self._epochs:
            source_train_input, source_train_task, source_train_embedding = source_train_set.next_batch(batch_size)
            source_train_task = np.reshape(source_train_task, [-1, 1])
            target_train_input, target_train_task, target_train_embedding = target_train_set.next_batch(batch_size)
            # 损失函数
            loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss, loss_of_reconstruction\
                = self._sess.run(
                (self._loss_of_whole_generator, self._loss_of_discriminator,
                 self._loss_classification, self._gen_loss, self._loss_of_reconstruction),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob,
                           self._source_xk: source_train_embedding, self._target_xk: target_train_embedding})

            # 最小化训练
            self._sess.run(
                (self._generator_train_op, self._clf_train_op, self._discriminator_train_op),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob,
                           self._source_xk: source_train_embedding, self._target_xk: target_train_embedding})
            if self._output_n_epoch < source_train_set.epoch_completed:

                source_test_prediction = self._sess.run(
                    self._source_prognosis_predict,
                    feed_dict={self._source_input: source_test_set.x, self._keep_prob: 1,
                               self._source_xk: source_test_set.xk})

                # source_test_auc = roc_auc_score(source_test_set.y, source_test_prediction)
                target_test_prediction = self._sess.run(
                    self._target_prognosis_predict,
                    feed_dict={self._target_input: target_test_set.x, self._keep_prob: 1,
                               self._target_xk: target_test_set.xk})
                target_test_auc, target_test_precision, target_test_recall, target_test_f_score, target_test_accuracy \
                    = calculate_score(target_test_set.y, target_test_prediction)
                source_test_auc,  source_test_precision, source_test_recall, source_test_f_score, source_test_accuracy \
                    = calculate_score(source_test_set.y, source_test_prediction)
                self._output_n_epoch = self._output_n_epoch + 1
                train_data.append([source_test_auc, source_test_precision, source_test_recall, source_test_f_score,
                                   source_test_accuracy, target_test_auc, target_test_precision, target_test_recall,
                                   target_test_f_score, target_test_accuracy, source_test_auc + target_test_auc])
                print("epoch:{}\tsource_test_auc:{}\ttarget_test_auc:{}\tloss_of_whole_generator:{}\t"
                      "loss_of_discriminator:{}\tloss_classification:{}\tgen_loss:{}\tloss_of_reconstruction:{}\t".
                      format(source_train_set.epoch_completed, source_test_auc, target_test_auc,
                             loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss,
                             loss_of_reconstruction))

        train_data = np.array(train_data)
        max_i = np.where(train_data[:, -1] == train_data[:, -1].max())
        print(np.squeeze(train_data[max_i, :]))
        return np.squeeze(train_data[max_i, :])

    @staticmethod
    def _hidden_layer_of_prognosis(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 10, activation=tf.nn.tanh, name="first_hidden_output")
        # first_hidden_output = \
        #     tf.contrib.layers.fully_connected(tensor_input, 5, activation_fn=tf.identity, name='first_hidden_output')
        prognosis_output = tf.layers.dense(first_hidden_output, 1, name=name)
        prognosis_predict = tf.nn.sigmoid(prognosis_output)
        return prognosis_output, prognosis_predict

    @staticmethod
    def _hidden_layer_of_discriminator(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 10, activation=tf.nn.tanh, name="first_hidden_output")
        discriminator_output = tf.layers.dense(first_hidden_output, 1, name=name)
        discriminator_predict = tf.nn.sigmoid(discriminator_output)
        return discriminator_output, discriminator_predict


class KAMA_V2(object):
    def __init__(self, n_input, feature_encoder_hidden, feature_embedding_input, n_class,
                 transfer=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer(),
                 epochs=300, name='kama', sess=None):
        """
        a Knowledge-Aware Multi-center clinical dataset Adaptation model Variant 2
        Variant 2 removal of pai_k from clinical center-discriminator

        :param n_input: num of input node
        :param feature_encoder_hidden: list of hidden feature encoder layer node num
        :param n_class: num of output node
        :param optimizer: optimization algorithm
        :param epochs: training epoch
        :param name: model name
        :param sess: tf.sess
        :param
        """
        self._n_input = n_input
        self._feature_encoder_hidden = feature_encoder_hidden
        self._feature_embedding_input = feature_embedding_input
        self._n_class = n_class
        self._epochs = epochs
        self._name = name

        with tf.variable_scope(self._name):
            self._transfer_func = transfer
            self._optimizer = optimizer
            # output_n_epoch: now training epoch
            self._output_n_epoch = 0
            self._sess = sess if sess is not None else tf.Session()
            self._graph_definition()

    def _u_init(self):
        with tf.variable_scope("u"):
            u = dict()
            u['up'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="up")
            u['uk'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="uk")
            return u

    def _attention_definition(self, zp, zk):
        source_correlation_matrix = tf.matmul(tf.transpose(zp), tf.nn.tanh(zk))
        self._ap = tf.reshape(tf.nn.tanh(tf.reduce_mean(source_correlation_matrix, axis=1)), [1, -1])
        self._ak = tf.reshape(tf.nn.tanh(tf.reduce_mean(source_correlation_matrix, axis=0)), [1, -1])
        ip = tf.ones_like(zp[:, 0:1])
        ik = tf.ones_like(zk[:, 0:1])
        self._bp = tf.nn.tanh((zp + (ip @ self._ap) * zk) @ self._u['up'])
        self._bk = tf.nn.tanh((zk + (ik @ self._ak) * zp) @ self._u['uk'])
        pai_p = self._bp * zp
        pai_k = self._bk * zk
        return pai_p, tf.concat([pai_p, pai_k], axis=1)

    def _weights_init(self):
        with tf.variable_scope("dae_weights"):
            weights = dict()
            weights['w1'] = tf.Variable(xavier_init(self._feature_embedding_input,
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="w1")
            weights['b1'] = tf.Variable(tf.zeros(self._feature_encoder_hidden[-1]), name="b1")
            weights['w2'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_embedding_input), dtype=tf.float32, name="w2")
            weights['b2'] = tf.Variable(tf.zeros(self._feature_embedding_input), name="b2")
            return weights

    def _graph_definition(self):
        self._source_input = tf.placeholder(tf.float32, [None, self._n_input], name="source_input")
        self._target_input = tf.placeholder(tf.float32, [None, self._n_input], name="target_input")
        self._source_label = tf.placeholder(tf.float32, [None, self._n_class], name="source_label")
        self._target_label = tf.placeholder(tf.float32, [None, self._n_class], name="target_label")

        self._source_xk = tf.placeholder(tf.float32, [None, self._feature_embedding_input], name="source_xk")
        self._target_xk = tf.placeholder(tf.float32, [None, self._feature_embedding_input], name="target_xk")
        self._keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("generator"):
            # knowledge feature extraction
            self._dae_weights = self._weights_init()
            # knowledge feature encoder
            self._source_zk = self._source_xk @ self._dae_weights['w1'] + self._dae_weights['b1']
            self._target_zk = self._target_xk @ self._dae_weights['w1'] + self._dae_weights['b1']
            # knowledge feature decoder
            self._source_reconstruction = self._source_zk @ self._dae_weights['w2'] + self._dae_weights['b2']
            self._target_reconstruction = self._target_zk @ self._dae_weights['w2'] + self._dae_weights['b2']

            self._loss_of_reconstruction = tf.reduce_mean(
                tf.losses.mean_squared_error(self._source_xk, self._source_reconstruction)) + tf.reduce_mean(
                tf.losses.mean_squared_error(self._target_xk, self._target_reconstruction))
            # tensor after feature encoder
            with tf.variable_scope('feature_encoder', reuse=tf.AUTO_REUSE):
                self._source_feature_encoder_output = self._init_feature_encoder(self._source_input)
                self._target_feature_encoder_output = self._init_feature_encoder(self._target_input)
            # attention
            # correlation_matrix M
            with tf.variable_scope('knowledge-aware_patient_representation', reuse=tf.AUTO_REUSE):
                self._u = self._u_init()
                self._source_patient_representation_pai_p, self._source_patient_representation\
                    = self._attention_definition(self._source_feature_encoder_output, self._source_zk)
                self._target_patient_representation_pai_p, self._target_patient_representation \
                    = self._attention_definition(self._target_feature_encoder_output, self._target_zk)
                # prediction of two data set
        with tf.variable_scope("prognosis", reuse=tf.AUTO_REUSE):
                self._source_prognosis_output, self._source_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._source_patient_representation, name="prognosis")
                self._target_prognosis_output, self._target_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._target_patient_representation, name="prognosis")
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            self._source_discriminator_output, self._source_discriminator_predict = self._hidden_layer_of_discriminator(
                self._source_patient_representation_pai_p, name="discriminator")
            self._target_discriminator_output, self._target_discriminator_predict = self._hidden_layer_of_discriminator(
                self._target_patient_representation_pai_p, name="discriminator")

        # discriminator loss
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._target_discriminator_output,
                                                    labels=tf.zeros_like(self._target_discriminator_output)))
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._source_discriminator_output,
                                                    labels=tf.ones_like(self._source_discriminator_output)))
        self._loss_of_discriminator = tf.add(fake_loss, real_loss)
        # generator loss
        self._gen_loss = self._loss_of_discriminator
        self._loss_classification = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self._source_label, logits=self._source_prognosis_output))
        self._loss_of_whole_generator = self._loss_classification - 0.5 * self._gen_loss + self._loss_of_reconstruction
        self._loss_of_whole_discriminator = self._loss_of_discriminator

        train_vars = tf.trainable_variables()
        # 分类器变量
        clf_vars = [var for var in train_vars if var.name.startswith('kama/prognosis')]
        print(clf_vars)
        # 生成器变量
        gen_vars = [var for var in train_vars if var.name.startswith('kama/generator')]
        print(gen_vars)
        # 判别器变量
        dis_vars = [var for var in train_vars if var.name.startswith('kama/discriminator')]
        print(dis_vars)
        self._generator_train_op = self._optimizer.minimize(self._loss_of_whole_generator, var_list=gen_vars)
        self._discriminator_train_op = self._optimizer.minimize(self._loss_of_discriminator, var_list=dis_vars)
        self._clf_train_op = self._optimizer.minimize(self._loss_classification, var_list=clf_vars)

    def _init_feature_encoder(self, input_data):
        hidden_outputs = []
        for i in range(len(self._feature_encoder_hidden)):
            if i == 0:
                hidden_output = tf.layers.dense(input_data, self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.tanh)

            else:
                hidden_output = tf.layers.dense(hidden_outputs[i-1], self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.tanh)

            hidden_outputs.append(hidden_output)
        embedding_output = tf.layers.dropout(inputs=hidden_outputs[-1], rate=1-self._keep_prob)
        return embedding_output

    def fit(self, source_train_set, source_test_set, target_train_set, target_test_set, batch_size, keep_prob):
        train_data = []
        self._sess.run(tf.global_variables_initializer())
        for c in tf.trainable_variables(self._name):
            print(c.name)
        while source_train_set.epoch_completed < self._epochs:
            source_train_input, source_train_task, source_train_embedding = source_train_set.next_batch(batch_size)
            source_train_task = np.reshape(source_train_task, [-1, 1])
            target_train_input, target_train_task, target_train_embedding = target_train_set.next_batch(batch_size)
            # 损失函数
            loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss, loss_of_reconstruction\
                = self._sess.run(
                (self._loss_of_whole_generator, self._loss_of_discriminator,
                 self._loss_classification, self._gen_loss, self._loss_of_reconstruction),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob,
                           self._source_xk: source_train_embedding, self._target_xk: target_train_embedding})
            # 最小化训练
            self._sess.run(
                (self._generator_train_op, self._clf_train_op, self._discriminator_train_op),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob,
                           self._source_xk: source_train_embedding, self._target_xk: target_train_embedding})

            if self._output_n_epoch < source_train_set.epoch_completed:

                source_test_prediction = self._sess.run(
                    self._source_prognosis_predict,
                    feed_dict={self._source_input: source_test_set.x, self._keep_prob: 1,
                               self._source_xk: source_test_set.xk})

                target_test_prediction = self._sess.run(
                    self._target_prognosis_predict,
                    feed_dict={self._target_input: target_test_set.x, self._keep_prob: 1,
                               self._target_xk: target_test_set.xk})

                target_test_auc, target_test_precision, target_test_recall, target_test_f_score, target_test_accuracy \
                    = calculate_score(target_test_set.y, target_test_prediction)
                source_test_auc,  source_test_precision, source_test_recall, source_test_f_score, source_test_accuracy \
                    = calculate_score(source_test_set.y, source_test_prediction)
                self._output_n_epoch = self._output_n_epoch + 1
                train_data.append([source_test_auc, source_test_precision, source_test_recall, source_test_f_score,
                                   source_test_accuracy, target_test_auc, target_test_precision, target_test_recall,
                                   target_test_f_score, target_test_accuracy, source_test_auc + target_test_auc])
                print("epoch:{}\tsource_test_auc:{}\ttarget_test_auc:{}\tloss_of_whole_generator:{}\t"
                      "loss_of_discriminator:{}\tloss_classification:{}\tgen_loss:{}\tloss_of_reconstruction:{}\t".
                      format(source_train_set.epoch_completed, source_test_auc, target_test_auc,
                             loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss,
                             loss_of_reconstruction))

        train_data = np.array(train_data)
        max_i = np.where(train_data[:, -1] == train_data[:, -1].max())
        print(np.squeeze(train_data[max_i, :]))
        return np.squeeze(train_data[max_i, :])

    @staticmethod
    def _hidden_layer_of_prognosis(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 10, activation=tf.nn.tanh, name="first_hidden_output")
        prognosis_output = tf.layers.dense(first_hidden_output, 1, name=name)
        prognosis_predict = tf.nn.sigmoid(prognosis_output)
        return prognosis_output, prognosis_predict

    @staticmethod
    def _hidden_layer_of_discriminator(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 10, activation=tf.nn.tanh, name="first_hidden_output")
        discriminator_output = tf.layers.dense(first_hidden_output, 1, name=name)
        discriminator_predict = tf.nn.sigmoid(discriminator_output)
        return discriminator_output, discriminator_predict


class KAMA_V3(object):
    def __init__(self, n_input, feature_encoder_hidden, feature_embedding_input, n_class,
                 transfer=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer(),
                 epochs=300, name='kama', sess=None):
        """
        a Knowledge-Aware Multi-center clinical dataset Adaptation model Variant 3
        Variant 3 removal of the knowledge-aware attention

        :param n_input: num of input node
        :param feature_encoder_hidden: list of hidden feature encoder layer node num
        :param n_class: num of output node
        :param optimizer: optimization algorithm
        :param epochs: training epoch
        :param name: model name
        :param sess: tf.sess
        :param
        """
        self._n_input = n_input
        self._feature_encoder_hidden = feature_encoder_hidden
        self._feature_embedding_input = feature_embedding_input
        self._n_class = n_class
        self._epochs = epochs
        self._name = name

        with tf.variable_scope(self._name):
            self._transfer_func = transfer

            self._optimizer = optimizer
            # output_n_epoch: now training epoch
            self._output_n_epoch = 0
            self._sess = sess if sess is not None else tf.Session()
            self._graph_definition()

    def _u_init(self):
        with tf.variable_scope("u"):
            u = dict()
            u['up'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="up")
            u['uk'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="uk")
            return u

    @staticmethod
    def _attention_definition(zp, zk):
        return tf.concat([zp, zk], axis=1)

    def _weights_init(self):
        with tf.variable_scope("dae_weights"):
            weights = dict()
            weights['w1'] = tf.Variable(xavier_init(self._feature_embedding_input,
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="w1")
            weights['b1'] = tf.Variable(tf.zeros(self._feature_encoder_hidden[-1]), name="b1")
            weights['w2'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_embedding_input), dtype=tf.float32, name="w2")
            weights['b2'] = tf.Variable(tf.zeros(self._feature_embedding_input), name="b2")
            return weights

    def _graph_definition(self):
        self._source_input = tf.placeholder(tf.float32, [None, self._n_input], name="source_input")
        self._target_input = tf.placeholder(tf.float32, [None, self._n_input], name="target_input")
        self._source_label = tf.placeholder(tf.float32, [None, self._n_class], name="source_label")
        self._target_label = tf.placeholder(tf.float32, [None, self._n_class], name="target_label")

        self._source_xk = tf.placeholder(tf.float32, [None, self._feature_embedding_input], name="source_xk")
        self._target_xk = tf.placeholder(tf.float32, [None, self._feature_embedding_input], name="target_xk")
        self._keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("generator"):
            # knowledge feature extraction
            self._dae_weights = self._weights_init()
            # knowledge feature encoder
            self._source_zk = self._source_xk @ self._dae_weights['w1'] + self._dae_weights['b1']
            self._target_zk = self._target_xk @ self._dae_weights['w1'] + self._dae_weights['b1']
            # knowledge feature decoder
            self._source_reconstruction = self._source_zk @ self._dae_weights['w2'] + self._dae_weights['b2']
            self._target_reconstruction = self._target_zk @ self._dae_weights['w2'] + self._dae_weights['b2']

            self._loss_of_reconstruction = tf.reduce_mean(
                tf.losses.mean_squared_error(self._source_xk, self._source_reconstruction)) + tf.reduce_mean(
                tf.losses.mean_squared_error(self._target_xk, self._target_reconstruction))
            # tensor after feature encoder
            with tf.variable_scope('feature_encoder', reuse=tf.AUTO_REUSE):
                self._source_feature_encoder_output = self._init_feature_encoder(self._source_input)
                self._target_feature_encoder_output = self._init_feature_encoder(self._target_input)
            # attention
            # correlation_matrix M
            with tf.variable_scope('knowledge-aware_patient_representation', reuse=tf.AUTO_REUSE):
                self._u = self._u_init()
                self._source_patient_representation = self._attention_definition(self._source_feature_encoder_output,
                                                                                 self._source_zk)
                self._target_patient_representation = self._attention_definition(self._target_feature_encoder_output,
                                                                                 self._target_zk)
                # prediction of two data set
        with tf.variable_scope("prognosis", reuse=tf.AUTO_REUSE):
                self._source_prognosis_output, self._source_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._source_patient_representation, name="prognosis")
                self._target_prognosis_output, self._target_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._target_patient_representation, name="prognosis")
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            self._source_discriminator_output, self._source_discriminator_predict = self._hidden_layer_of_discriminator(
                self._source_patient_representation, name="discriminator")
            self._target_discriminator_output, self._target_discriminator_predict = self._hidden_layer_of_discriminator(
                self._target_patient_representation, name="discriminator")

        # discriminator loss
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._target_discriminator_output,
                                                    labels=tf.zeros_like(self._target_discriminator_output)))
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._source_discriminator_output,
                                                    labels=tf.ones_like(self._source_discriminator_output)))
        self._loss_of_discriminator = tf.add(fake_loss, real_loss)
        # generator loss
        self._gen_loss = self._loss_of_discriminator
        self._loss_classification = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self._source_label, logits=self._source_prognosis_output))
        self._loss_of_whole_generator = self._loss_classification - 0.5 * self._gen_loss + self._loss_of_reconstruction
        self._loss_of_whole_discriminator = self._loss_of_discriminator

        train_vars = tf.trainable_variables()
        # 分类器变量
        clf_vars = [var for var in train_vars if var.name.startswith('kama/prognosis')]
        print(clf_vars)
        # 生成器变量
        gen_vars = [var for var in train_vars if var.name.startswith('kama/generator')]
        print(gen_vars)
        # 判别器变量
        dis_vars = [var for var in train_vars if var.name.startswith('kama/discriminator')]
        print(dis_vars)
        # self._generator_train_op = self._optimizer.minimize(self._loss_of_whole_generator)
        self._generator_train_op = self._optimizer.minimize(self._loss_of_whole_generator, var_list=gen_vars)
        self._discriminator_train_op = self._optimizer.minimize(self._loss_of_discriminator, var_list=dis_vars)
        self._clf_train_op = self._optimizer.minimize(self._loss_classification, var_list=clf_vars)

    def _init_feature_encoder(self, input_data):
        hidden_outputs = []
        for i in range(len(self._feature_encoder_hidden)):
            if i == 0:
                hidden_output = tf.layers.dense(input_data, self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.tanh)

            else:
                hidden_output = tf.layers.dense(hidden_outputs[i-1], self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.tanh)

            hidden_outputs.append(hidden_output)
        embedding_output = tf.layers.dropout(inputs=hidden_outputs[-1], rate=1-self._keep_prob)
        return embedding_output

    def fit(self, source_train_set, source_test_set, target_train_set, target_test_set, batch_size, keep_prob):
        train_data = []
        self._sess.run(tf.global_variables_initializer())
        for c in tf.trainable_variables(self._name):
            print(c.name)

        while source_train_set.epoch_completed < self._epochs:
            source_train_input, source_train_task, source_train_embedding = source_train_set.next_batch(batch_size)
            source_train_task = np.reshape(source_train_task, [-1, 1])
            target_train_input, target_train_task, target_train_embedding = target_train_set.next_batch(batch_size)

            # 损失函数
            loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss, loss_of_reconstruction\
                = self._sess.run(
                (self._loss_of_whole_generator, self._loss_of_discriminator,
                 self._loss_classification, self._gen_loss, self._loss_of_reconstruction),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob,
                           self._source_xk: source_train_embedding, self._target_xk: target_train_embedding})
            # 最小化训练
            self._sess.run(
                (self._generator_train_op, self._clf_train_op, self._discriminator_train_op),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob,
                           self._source_xk: source_train_embedding, self._target_xk: target_train_embedding})

            if self._output_n_epoch < source_train_set.epoch_completed:

                source_test_prediction = self._sess.run(
                    self._source_prognosis_predict,
                    feed_dict={self._source_input: source_test_set.x, self._keep_prob: 1,
                               self._source_xk: source_test_set.xk})

                # source_test_auc = roc_auc_score(source_test_set.y, source_test_prediction)
                target_test_prediction = self._sess.run(
                    self._target_prognosis_predict,
                    feed_dict={self._target_input: target_test_set.x, self._keep_prob: 1,
                               self._target_xk: target_test_set.xk})

                target_test_auc, target_test_precision, target_test_recall, target_test_f_score, target_test_accuracy \
                    = calculate_score(target_test_set.y, target_test_prediction)
                source_test_auc,  source_test_precision, source_test_recall, source_test_f_score, source_test_accuracy \
                    = calculate_score(source_test_set.y, source_test_prediction)
                self._output_n_epoch = self._output_n_epoch + 1
                train_data.append([source_test_auc, source_test_precision, source_test_recall, source_test_f_score,
                                   source_test_accuracy, target_test_auc, target_test_precision, target_test_recall,
                                   target_test_f_score, target_test_accuracy, source_test_auc + target_test_auc])
                print("epoch:{}\tsource_test_auc:{}\ttarget_test_auc:{}\tloss_of_whole_generator:{}\t"
                      "loss_of_discriminator:{}\tloss_classification:{}\tgen_loss:{}\tloss_of_reconstruction:{}\t".
                      format(source_train_set.epoch_completed, source_test_auc, target_test_auc,
                             loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss,
                             loss_of_reconstruction))

        train_data = np.array(train_data)
        max_i = np.where(train_data[:, -1] == train_data[:, -1].max())
        print(np.squeeze(train_data[max_i, :]))
        return np.squeeze(train_data[max_i, :])

    @staticmethod
    def _hidden_layer_of_prognosis(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 10, activation=tf.nn.tanh, name="first_hidden_output")
        prognosis_output = tf.layers.dense(first_hidden_output, 1, name=name)
        prognosis_predict = tf.nn.sigmoid(prognosis_output)
        return prognosis_output, prognosis_predict

    @staticmethod
    def _hidden_layer_of_discriminator(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 10, activation=tf.nn.tanh, name="first_hidden_output")
        discriminator_output = tf.layers.dense(first_hidden_output, 1, name=name)
        discriminator_predict = tf.nn.sigmoid(discriminator_output)
        return discriminator_output, discriminator_predict


class KAMA_V4(object):
    def __init__(self, n_input, feature_encoder_hidden, feature_embedding_input, n_class,
                 transfer=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer(),
                 epochs=300, name='kama', sess=None):
        """
        a Knowledge-Aware Multi-center clinical dataset Adaptation model Variant 4
        Variant 4 removal of pai_k from clinical outcome predictor

        :param n_input: num of input node
        :param feature_encoder_hidden: list of hidden feature encoder layer node num
        :param n_class: num of output node
        :param optimizer: optimization algorithm
        :param epochs: training epoch
        :param name: model name
        :param sess: tf.sess
        :param
        """
        self._n_input = n_input
        self._feature_encoder_hidden = feature_encoder_hidden
        self._feature_embedding_input = feature_embedding_input
        self._n_class = n_class
        self._epochs = epochs
        self._name = name

        with tf.variable_scope(self._name):
            self._transfer_func = transfer
            self._optimizer = optimizer
            # output_n_epoch: now training epoch
            self._output_n_epoch = 0
            self._sess = sess if sess is not None else tf.Session()
            self._graph_definition()

    def _u_init(self):
        with tf.variable_scope("u"):
            u = dict()
            u['up'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="up")
            u['uk'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="uk")
            return u

    def _attention_definition(self, zp, zk):
        source_correlation_matrix = tf.matmul(tf.transpose(zp), tf.nn.tanh(zk))
        self._ap = tf.reshape(tf.nn.tanh(tf.reduce_mean(source_correlation_matrix, axis=1)), [1, -1])
        self._ak = tf.reshape(tf.nn.tanh(tf.reduce_mean(source_correlation_matrix, axis=0)), [1, -1])
        ip = tf.ones_like(zp[:, 0:1])
        ik = tf.ones_like(zk[:, 0:1])
        self._bp = tf.nn.tanh((zp + (ip @ self._ap) * zk) @ self._u['up'])
        self._bk = tf.nn.tanh((zk + (ik @ self._ak) * zp) @ self._u['uk'])
        pai_p = self._bp * zp
        pai_k = self._bk * zk
        return pai_p, tf.concat([pai_p, pai_k], axis=1)

    def _weights_init(self):
        with tf.variable_scope("dae_weights"):
            weights = dict()
            weights['w1'] = tf.Variable(xavier_init(self._feature_embedding_input,
                                                    self._feature_encoder_hidden[-1]), dtype=tf.float32, name="w1")
            weights['b1'] = tf.Variable(tf.zeros(self._feature_encoder_hidden[-1]), name="b1")
            weights['w2'] = tf.Variable(xavier_init(self._feature_encoder_hidden[-1],
                                                    self._feature_embedding_input), dtype=tf.float32, name="w2")
            weights['b2'] = tf.Variable(tf.zeros(self._feature_embedding_input), name="b2")
            return weights

    def _graph_definition(self):
        self._source_input = tf.placeholder(tf.float32, [None, self._n_input], name="source_input")
        self._target_input = tf.placeholder(tf.float32, [None, self._n_input], name="target_input")
        self._source_label = tf.placeholder(tf.float32, [None, self._n_class], name="source_label")
        self._target_label = tf.placeholder(tf.float32, [None, self._n_class], name="target_label")

        self._source_xk = tf.placeholder(tf.float32, [None, self._feature_embedding_input], name="source_xk")
        self._target_xk = tf.placeholder(tf.float32, [None, self._feature_embedding_input], name="target_xk")
        self._keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("generator"):
            # knowledge feature extraction
            self._dae_weights = self._weights_init()
            # knowledge feature encoder
            self._source_zk = self._source_xk @ self._dae_weights['w1'] + self._dae_weights['b1']
            self._target_zk = self._target_xk @ self._dae_weights['w1'] + self._dae_weights['b1']
            # knowledge feature decoder
            self._source_reconstruction = self._source_zk @ self._dae_weights['w2'] + self._dae_weights['b2']
            self._target_reconstruction = self._target_zk @ self._dae_weights['w2'] + self._dae_weights['b2']

            self._loss_of_reconstruction = tf.reduce_mean(
                tf.losses.mean_squared_error(self._source_xk, self._source_reconstruction)) + tf.reduce_mean(
                tf.losses.mean_squared_error(self._target_xk, self._target_reconstruction))
            # tensor after feature encoder
            with tf.variable_scope('feature_encoder', reuse=tf.AUTO_REUSE):
                self._source_feature_encoder_output = self._init_feature_encoder(self._source_input)
                self._target_feature_encoder_output = self._init_feature_encoder(self._target_input)
            # attention
            # correlation_matrix M
            with tf.variable_scope('knowledge-aware_patient_representation', reuse=tf.AUTO_REUSE):
                self._u = self._u_init()
                self._source_patient_representation_pai_p, self._source_patient_representation = self._attention_definition(
                    self._source_feature_encoder_output, self._source_zk)
                self._target_patient_representation_pai_p, self._target_patient_representation = self._attention_definition(
                    self._target_feature_encoder_output, self._target_zk)
                # prediction of two data set
        with tf.variable_scope("prognosis", reuse=tf.AUTO_REUSE):
                self._source_prognosis_output, self._source_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._source_patient_representation_pai_p, name="prognosis")
                self._target_prognosis_output, self._target_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._target_patient_representation_pai_p, name="prognosis")
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            self._source_discriminator_output, self._source_discriminator_predict = self._hidden_layer_of_discriminator(
                self._source_patient_representation, name="discriminator")
            self._target_discriminator_output, self._target_discriminator_predict = self._hidden_layer_of_discriminator(
                self._target_patient_representation, name="discriminator")

        # discriminator loss
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._target_discriminator_output,
                                                    labels=tf.zeros_like(self._target_discriminator_output)))
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._source_discriminator_output,
                                                    labels=tf.ones_like(self._source_discriminator_output)))
        self._loss_of_discriminator = tf.add(fake_loss, real_loss)
        # generator loss
        self._gen_loss = self._loss_of_discriminator
        self._loss_classification = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self._source_label, logits=self._source_prognosis_output))
        self._loss_of_whole_generator = self._loss_classification - 0.5 * self._gen_loss + self._loss_of_reconstruction
        self._loss_of_whole_discriminator = self._loss_of_discriminator

        train_vars = tf.trainable_variables()
        # 分类器变量
        clf_vars = [var for var in train_vars if var.name.startswith('kama/prognosis')]
        print(clf_vars)
        # 生成器变量
        gen_vars = [var for var in train_vars if var.name.startswith('kama/generator')]
        print(gen_vars)
        # 判别器变量
        dis_vars = [var for var in train_vars if var.name.startswith('kama/discriminator')]
        print(dis_vars)
        # self._generator_train_op = self._optimizer.minimize(self._loss_of_whole_generator)
        self._generator_train_op = self._optimizer.minimize(self._loss_of_whole_generator, var_list=gen_vars)
        self._discriminator_train_op = self._optimizer.minimize(self._loss_of_discriminator, var_list=dis_vars)
        self._clf_train_op = self._optimizer.minimize(self._loss_classification, var_list=clf_vars)

    def _init_feature_encoder(self, input_data):
        hidden_outputs = []
        for i in range(len(self._feature_encoder_hidden)):
            if i == 0:
                hidden_output = tf.layers.dense(input_data, self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.tanh)

            else:
                hidden_output = tf.layers.dense(hidden_outputs[i-1], self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.tanh)

            hidden_outputs.append(hidden_output)
        embedding_output = tf.layers.dropout(inputs=hidden_outputs[-1], rate=1-self._keep_prob)
        return embedding_output

    def fit(self, source_train_set, source_test_set, target_train_set, target_test_set, batch_size, keep_prob):
        train_data = []
        self._sess.run(tf.global_variables_initializer())

        for c in tf.trainable_variables(self._name):
            print(c.name)

        while source_train_set.epoch_completed < self._epochs:
            source_train_input, source_train_task, source_train_embedding = source_train_set.next_batch(batch_size)
            source_train_task = np.reshape(source_train_task, [-1, 1])
            target_train_input, target_train_task, target_train_embedding = target_train_set.next_batch(batch_size)

            # 损失函数
            loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss, loss_of_reconstruction\
                = self._sess.run(
                (self._loss_of_whole_generator, self._loss_of_discriminator,
                 self._loss_classification, self._gen_loss, self._loss_of_reconstruction),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob,
                           self._source_xk: source_train_embedding, self._target_xk: target_train_embedding})

            # 最小化训练
            self._sess.run(
                (self._generator_train_op, self._clf_train_op, self._discriminator_train_op),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob,
                           self._source_xk: source_train_embedding, self._target_xk: target_train_embedding})

            if self._output_n_epoch < source_train_set.epoch_completed:

                source_test_prediction = self._sess.run(
                    self._source_prognosis_predict,
                    feed_dict={self._source_input: source_test_set.x, self._keep_prob: 1,
                               self._source_xk: source_test_set.xk})

                target_test_prediction = self._sess.run(
                    self._target_prognosis_predict,
                    feed_dict={self._target_input: target_test_set.x, self._keep_prob: 1,
                               self._target_xk: target_test_set.xk})

                target_test_auc, target_test_precision, target_test_recall, target_test_f_score, target_test_accuracy \
                    = calculate_score(target_test_set.y, target_test_prediction)
                source_test_auc,  source_test_precision, source_test_recall, source_test_f_score, source_test_accuracy \
                    = calculate_score(source_test_set.y, source_test_prediction)
                self._output_n_epoch = self._output_n_epoch + 1
                train_data.append([source_test_auc, source_test_precision, source_test_recall, source_test_f_score,
                                   source_test_accuracy, target_test_auc, target_test_precision, target_test_recall,
                                   target_test_f_score, target_test_accuracy, source_test_auc + target_test_auc])
                print("epoch:{}\tsource_test_auc:{}\ttarget_test_auc:{}\tloss_of_whole_generator:{}\t"
                      "loss_of_discriminator:{}\tloss_classification:{}\tgen_loss:{}\tloss_of_reconstruction:{}\t".
                      format(source_train_set.epoch_completed, source_test_auc, target_test_auc,
                             loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss,
                             loss_of_reconstruction))

        train_data = np.array(train_data)
        max_i = np.where(train_data[:, -1] == train_data[:, -1].max())
        print(np.squeeze(train_data[max_i, :]))
        return np.squeeze(train_data[max_i, :])

    @staticmethod
    def _hidden_layer_of_prognosis(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 10, activation=tf.nn.tanh, name="first_hidden_output")
        prognosis_output = tf.layers.dense(first_hidden_output, 1, name=name)
        prognosis_predict = tf.nn.sigmoid(prognosis_output)
        return prognosis_output, prognosis_predict

    @staticmethod
    def _hidden_layer_of_discriminator(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 10, activation=tf.nn.tanh, name="first_hidden_output")
        discriminator_output = tf.layers.dense(first_hidden_output, 1, name=name)
        discriminator_predict = tf.nn.sigmoid(discriminator_output)
        return discriminator_output, discriminator_predict

