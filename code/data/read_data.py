import numpy as np


class DataSet(object):

    """
    输入特征与标签作为一个集合，可以用于小批次训练

    """
    def __init__(self, x, y, xk):
        self._x = x
        self._y = y
        self._xk = xk
        self._num_example = self._x.shape[0]
        self._index = np.arange(self._num_example)
        self._epoch_completed = 0
        self._index_in_epoch = 0
        if self._x.shape[0] != self._y.shape[0]:
            raise ValueError('The num example of x is not equal to y ')

    def next_batch(self, batch_size):
        # batch_size 输入标准未考虑
        if batch_size < 0 or batch_size > self._num_example:
            raise ValueError('The size of one batch: {} should be less than the total number of '
                             'data: {}'.format(batch_size, self.num_examples))
            # batch_size = self._num_example
            # self._shuffle()
        start = self._index_in_epoch
        if start + batch_size > self._num_example:
            self._index_in_epoch = self._num_example
            x_batch_rest = self._x[start:self._index_in_epoch]
            y_batch_rest = self._y[start:self._index_in_epoch]
            xk_batch_rest = self._xk[start:self._index_in_epoch]
            self._epoch_completed += 1
            self._index_in_epoch = 0
            self._shuffle()
            rest = start + batch_size - self._num_example
            x_batch_new = self._x[self._index_in_epoch:self._index_in_epoch + rest]
            y_batch_new = self._y[self._index_in_epoch:self._index_in_epoch + rest]
            xk_batch_new = self._xk[self._index_in_epoch:self._index_in_epoch + rest]
            self._index_in_epoch += rest

            return np.concatenate((x_batch_rest, x_batch_new), axis=0), \
                   np.concatenate((y_batch_rest, y_batch_new), axis=0), \
                   np.concatenate((xk_batch_rest, xk_batch_new), axis=0)
        else:
            self._index_in_epoch = start + batch_size
            x_batch = self._x[start:self._index_in_epoch]
            y_batch = self._y[start:self._index_in_epoch]
            xk_batch = self._xk[start:self._index_in_epoch]
            return x_batch, y_batch, xk_batch

    def _shuffle(self):
        index = np.arange(self._num_example)
        np.random.shuffle(index)
        self._index = index
        self._x = self._x[index]
        self._y = self._y[index]
        self._xk = self._xk[index]

    @property
    def num_examples(self):
        return self._num_example

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def xk(self):
        return self._xk

    @property
    def index(self):
        return self._index


    @property
    def epoch_completed(self):
        return self._epoch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
        self._epoch_completed = value
