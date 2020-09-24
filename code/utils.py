import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve

# constant
SQRT_CONST = 1e-10
KNOWLEDGE_GRAPH_DATA_SHEET = ['node', 'relation']
KNOWLEDGE_GRAPH_DIR = ''
KNOWLEDGE_GRAPH_SOURCE_DATA_PATH = KNOWLEDGE_GRAPH_DIR+'知识图谱节点与关系.xlsx'
KNOWLEDGE_GRAPH_EMBEDDING_SAVE_DIR = ''
MIMIC_FEATURE_EMBEDDING_SAVE_PATH = KNOWLEDGE_GRAPH_DIR + 'result/mimic_feature_embedding.xlsx'
PLAGH_FEATURE_EMBEDDING_SAVE_PATH = KNOWLEDGE_GRAPH_DIR + 'result/plagh_feature_embedding.xlsx'
PLAGH_PATIENT_FEATURE_SAVE_PATH = ''
MIMIC_PATIENT_FEATURE_SAVE_PATH = ''
PLAGH_KNOWLEDGE_FEATURE_SAVE_PATH = ''
MIMIC_KNOWLEDGE_FEATURE_SAVE_PATH = ''


MIMIC = 'mimic'
PLAGH = 'plagh'

INDEX = 'index'
TYPE = 'type'
TARGET = 'target'
WHOLE = 'WHOLE'
SUPER = 'SUPER'
TREAT = 'TREAT'
CAUSE = 'CAUSE'
HINT = 'HINT'
BE = 'BE'
SELF_LOOP = 'SELF_LOOP'
RELATION_LIST = [SUPER, TREAT, CAUSE, HINT, SELF_LOOP]
RELATION_DICT = {SUPER: 1, TREAT: 2, CAUSE: 3, HINT: 4}


def calculate_score(y_label, y_prediction, print_flag=False):
    auc = roc_auc_score(y_label, y_prediction)
    fpr, tpr, thresholds = roc_curve(y_label, y_prediction)
    threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred_label = (y_prediction >= threshold) * 1
    precision = precision_score(y_label, y_pred_label)
    recall = recall_score(y_label, y_pred_label)
    f_score = f1_score(y_label, y_pred_label)
    accuracy = accuracy_score(y_label, y_pred_label)
    if print_flag:
        print('auc:{} precision:{} recall:{} f_score:{} accuracy:{}'.format(auc, precision, recall, f_score, accuracy))
    return auc, precision, recall, f_score, accuracy


def calculate_score_and_get_roc(y_label, y_prediction, print_flag=False):
    auc = roc_auc_score(y_label, y_prediction)
    fpr, tpr, thresholds = roc_curve(y_label, y_prediction)
    threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred_label = (y_prediction >= threshold) * 1
    precision = precision_score(y_label, y_pred_label)
    recall = recall_score(y_label, y_pred_label)
    f_score = f1_score(y_label, y_pred_label)
    accuracy = accuracy_score(y_label, y_pred_label)
    if print_flag:
        print('auc:{} precision:{} recall:{} f_score:{} accuracy:{}'.format(auc, precision, recall, f_score, accuracy))
    return auc, precision, recall, f_score, accuracy, fpr, tpr, thresholds, y_pred_label


def xavier_init(fan_in, fan_out, constant=1):
    """
    initializing variable using xavier
    :param fan_in: input code
    :param fan_out: output code
    :param constant: constant
    :return:
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D


def safe_sqrt(x, lbound=SQRT_CONST):
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

