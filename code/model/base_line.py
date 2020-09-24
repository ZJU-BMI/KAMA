from utils import *


class MCMLP(object):
    def __init__(self, n_input, feature_encoder_hidden, n_class,
                 transfer=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer(),
                 epochs=500, name='mlp', sess=None):
        """
        MLP of multi-centers

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

    def _graph_definition(self):
        self._source_input = tf.placeholder(tf.float32, [None, self._n_input], name="source_input")
        self._target_input = tf.placeholder(tf.float32, [None, self._n_input], name="target_input")
        self._source_label = tf.placeholder(tf.float32, [None, self._n_class], name="source_label")
        self._target_label = tf.placeholder(tf.float32, [None, self._n_class], name="target_label")
        self._keep_prob = tf.placeholder(tf.float32)
        with tf.variable_scope("generator"):
            # tensor after feature encoder
            with tf.variable_scope('feature_encoder', reuse=tf.AUTO_REUSE):
                self._source_feature_encoder_output = self._init_feature_encoder(self._source_input)
                self._target_feature_encoder_output = self._init_feature_encoder(self._target_input)
            # prediction of two data set
            with tf.variable_scope("prognosis", reuse=tf.AUTO_REUSE):
                self._source_prognosis_output, self._source_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._source_feature_encoder_output, name="prognosis")
                self._target_prognosis_output, self._target_prognosis_predict = self._hidden_layer_of_prognosis(
                    self._target_feature_encoder_output, name="prognosis")

        # generator loss
        self._loss_classification = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self._source_label, logits=self._source_prognosis_output))
        self._loss_of_whole_generator = self._loss_classification
        train_vars = tf.trainable_variables()
        # 生成器变量
        gen_vars = [var for var in train_vars if var.name.startswith('mcal/generator')]
        print(gen_vars)
        self._generator_train_op = self._optimizer.minimize(self._loss_of_whole_generator, var_list=gen_vars)

    def _init_feature_encoder(self, input_data):
        hidden_outputs = []
        for i in range(len(self._feature_encoder_hidden)):
            if i == 0:
                hidden_output = tf.layers.dense(input_data, self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.elu)
            else:
                hidden_output = tf.layers.dense(hidden_outputs[i - 1], self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.elu)
            hidden_outputs.append(hidden_output)
        embedding_output = tf.layers.dropout(inputs=hidden_outputs[-1], rate=1 - self._keep_prob)
        return embedding_output

    def fit(self, source_train_set, source_test_set, target_train_set, target_test_set, batch_size, keep_prob):
        self._sess.run(tf.global_variables_initializer())
        train_data = []
        while source_train_set.epoch_completed < self._epochs:
            source_train_input, source_train_task, _ = source_train_set.next_batch(batch_size)
            source_train_task = np.reshape(source_train_task, [-1, 1])
            target_train_input, target_train_task, _ = target_train_set.next_batch(batch_size)

            # 损失函数
            loss_of_whole_generator, loss_classification = self._sess.run(
                (self._loss_of_whole_generator, self._loss_classification),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob})

            # 最小化训练
            self._sess.run(
                self._generator_train_op,
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob})

            if self._output_n_epoch < source_train_set.epoch_completed:
                source_test_prediction = self.predict(source_test_set.x)
                target_test_prediction = self.predict(target_test_set.x)
                target_test_auc, target_test_precision, target_test_recall, target_test_f_score, target_test_accuracy, \
                target_test_fpr, target_test_tpr, target_test_thresholds, target_test_prediction_label \
                    = calculate_score_and_get_roc(target_test_set.y, target_test_prediction)
                source_test_auc, source_test_precision, source_test_recall, source_test_f_score, source_test_accuracy, \
                source_test_fpr, source_test_tpr, source_test_thresholds, source_test_prediction_label \
                    = calculate_score_and_get_roc(source_test_set.y, source_test_prediction)
                train_data.append([source_test_auc, source_test_precision, source_test_recall, source_test_f_score,
                                   source_test_accuracy, target_test_auc, target_test_precision, target_test_recall,
                                   target_test_f_score, target_test_accuracy, source_test_auc + target_test_auc])

                self._output_n_epoch = self._output_n_epoch + 1

                print("epoch:{}\tsource_test_auc:{}\ttarget_test_auc:{}\tloss_of_whole_generator:{}\t"
                      "loss_classification:{}\t".
                      format(source_train_set.epoch_completed, source_test_auc, target_test_auc,
                             loss_of_whole_generator, loss_classification))

        train_data = np.array(train_data)
        max_i = np.where(train_data[:, -1] == train_data[:, -1].max())
        print(np.squeeze(train_data[max_i, :]))
        return np.squeeze(train_data[max_i, :])

    @staticmethod
    def _hidden_layer_of_prognosis(tensor_input, name):
        first_hidden_output = tf.contrib.layers.fully_connected(tensor_input, 10, activation_fn=tf.identity)
        prognosis_output = tf.layers.dense(first_hidden_output, 1, name=name)
        prognosis_predict = tf.nn.sigmoid(prognosis_output)
        return prognosis_output, prognosis_predict

    def predict(self, x):
        return self._sess.run(
                    self._source_prognosis_output,
                    feed_dict={self._source_input: x, self._keep_prob: 1})


class MCAL(object):
    def __init__(self, n_input, feature_encoder_hidden, n_class,
                 transfer=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer(),
                 epochs=500, name='mcal', sess=None):
        """
        Adversarial learning from data of multi-centers

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

    def _graph_definition(self):
        self._source_input = tf.placeholder(tf.float32, [None, self._n_input], name="source_input")
        self._target_input = tf.placeholder(tf.float32, [None, self._n_input], name="target_input")
        self._source_label = tf.placeholder(tf.float32, [None, self._n_class], name="source_label")
        self._target_label = tf.placeholder(tf.float32, [None, self._n_class], name="target_label")
        self._keep_prob = tf.placeholder(tf.float32)
        with tf.variable_scope("generator"):
            # tensor after feature encoder
            with tf.variable_scope('feature_encoder', reuse=tf.AUTO_REUSE):
                self._source_feature_encoder_output = self._init_feature_encoder(self._source_input)
                self._target_feature_encoder_output = self._init_feature_encoder(self._target_input)
            # prediction of two data set
        with tf.variable_scope("prognosis", reuse=tf.AUTO_REUSE):
            self._source_prognosis_output, self._source_prognosis_predict = self._hidden_layer_of_prognosis(
                self._source_feature_encoder_output, name="prognosis")
            self._target_prognosis_output, self._target_prognosis_predict = self._hidden_layer_of_prognosis(
                self._target_feature_encoder_output, name="prognosis")
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            self._source_discriminator_output, self._source_discriminator_predict = self._hidden_layer_of_discriminator(
                self._source_feature_encoder_output, name="discriminator")
            self._target_discriminator_output, self._target_discriminator_predict = self._hidden_layer_of_discriminator(
                self._target_feature_encoder_output, name="discriminator")

        # discriminator loss
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._target_discriminator_output,
                                                    labels=tf.zeros_like(self._target_discriminator_output)))
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._source_discriminator_output,
                                                    labels=tf.ones_like(self._source_discriminator_output)))
        self._loss_of_discriminator = tf.add(fake_loss, real_loss)
        self._gen_loss = 0.5 * self._loss_of_discriminator
        self._loss_classification = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self._source_label, logits=self._source_prognosis_output))
        self._loss_of_whole_generator = self._loss_classification - self._gen_loss

        train_vars = tf.trainable_variables()
        # 生成器变量
        gen_vars = [var for var in train_vars if var.name.startswith('mcal/generator')]
        print(gen_vars)

        clf_vars = [var for var in train_vars if var.name.startswith('mcal/prognosis')]
        print(clf_vars)
        # 判别器变量
        dis_vars = [var for var in train_vars if var.name.startswith('mcal/discriminator')]
        print(dis_vars)
        self._generator_train_op = self._optimizer.minimize(self._loss_of_whole_generator, var_list=gen_vars)
        self._clf_train_op = self._optimizer.minimize(self._loss_classification, var_list=clf_vars)
        self._discriminator_train_op = self._optimizer.minimize(self._loss_of_discriminator, var_list=dis_vars)

    def _init_feature_encoder(self, input_data):

        hidden_outputs = []
        for i in range(len(self._feature_encoder_hidden)):
            if i == 0:
                hidden_output = tf.layers.dense(input_data, self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.elu)

            else:
                hidden_output = tf.layers.dense(hidden_outputs[i - 1], self._feature_encoder_hidden[i],
                                                name="dense{}".format(i), activation=tf.nn.elu)

            hidden_outputs.append(hidden_output)
        embedding_output = tf.layers.dropout(inputs=hidden_outputs[-1], rate=1 - self._keep_prob)
        return embedding_output

    def fit(self, source_train_set, source_test_set, target_train_set, target_test_set, batch_size, keep_prob):
        self._sess.run(tf.global_variables_initializer())
        train_data = []
        while source_train_set.epoch_completed < self._epochs:
            source_train_input, source_train_task, _ = source_train_set.next_batch(batch_size)
            source_train_task = np.reshape(source_train_task, [-1, 1])
            target_train_input, target_train_task, _ = target_train_set.next_batch(batch_size)
            # 损失函数
            loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss = self._sess.run(
                (self._loss_of_whole_generator, self._loss_of_discriminator, self._loss_classification, self._gen_loss),
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob})
            # 最小化训练
            self._sess.run(
                self._generator_train_op,
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob})
            self._sess.run(
                self._clf_train_op,
                feed_dict={self._source_input: source_train_input, self._target_input: target_train_input,
                           self._source_label: source_train_task, self._keep_prob: keep_prob})

            self._sess.run(self._discriminator_train_op, feed_dict={
                self._source_input: source_train_input, self._target_input: target_train_input,
                self._keep_prob: keep_prob})

            if self._output_n_epoch < source_train_set.epoch_completed:
                source_test_prediction = self.predict(source_test_set.x)
                target_test_prediction = self.predict(target_test_set.x)

                target_test_auc, target_test_precision, target_test_recall, target_test_f_score, target_test_accuracy, \
                target_test_fpr, target_test_tpr, target_test_thresholds, target_test_prediction_label \
                    = calculate_score_and_get_roc(target_test_set.y, target_test_prediction)
                source_test_auc, source_test_precision, source_test_recall, source_test_f_score, source_test_accuracy, \
                source_test_fpr, source_test_tpr, source_test_thresholds, source_test_prediction_label \
                    = calculate_score_and_get_roc(source_test_set.y, source_test_prediction)

                self._output_n_epoch = self._output_n_epoch + 1
                train_data.append([source_test_auc, source_test_precision, source_test_recall, source_test_f_score,
                                   source_test_accuracy, target_test_auc, target_test_precision, target_test_recall,
                                   target_test_f_score, target_test_accuracy, source_test_auc + target_test_auc])
                print("epoch:{}\tsource_test_auc:{}\ttarget_test_auc:{}\tloss_of_whole_generator:{}\t"
                      "loss_of_discriminator:{}\tloss_classification:{}\tgen_loss:{}\t".
                      format(source_train_set.epoch_completed, source_test_auc, target_test_auc,
                             loss_of_whole_generator, loss_of_discriminator, loss_classification, gen_loss))

        train_data = np.array(train_data)
        max_i = np.where(train_data[:, -1] == train_data[:, -1].max())
        print(np.squeeze(train_data[max_i, :]))
        return np.squeeze(train_data[max_i, :])

    def predict(self, x):
        return self._sess.run(
                    self._source_prognosis_output,
                    feed_dict={self._source_input: x, self._keep_prob: 1})

    @staticmethod
    def _hidden_layer_of_prognosis(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 5, activation=tf.nn.tanh, name="first_hidden_output")
        prognosis_output = tf.layers.dense(first_hidden_output, 1, name=name)
        prognosis_predict = tf.nn.sigmoid(prognosis_output)
        return prognosis_output, prognosis_predict

    @staticmethod
    def _hidden_layer_of_discriminator(tensor_input, name):
        first_hidden_output = tf.layers.dense(tensor_input, 5, activation=tf.nn.tanh, name="first_hidden_output")
        discriminator_output = tf.layers.dense(first_hidden_output, 1, name=name)
        discriminator_predict = tf.nn.sigmoid(discriminator_output)
        return discriminator_output, discriminator_predict


