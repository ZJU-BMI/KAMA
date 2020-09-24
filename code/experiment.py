import pandas as pd
import copy
from sklearn.model_selection import StratifiedShuffleSplit
from data.read_data import DataSet
from model.kama import *
from model.base_line import *
from knowledge_graph.knowledge_graph import KnowledgeGraph


def read_data_of_sjz(group):
    if group == 1:
        source_data = pd.read_excel(MIMIC_PATIENT_FEATURE_SAVE_PATH)
        target_data = pd.read_excel(PLAGH_PATIENT_FEATURE_SAVE_PATH)
    else:
        target_data = pd.read_excel(MIMIC_PATIENT_FEATURE_SAVE_PATH)
        source_data = pd.read_excel(PLAGH_PATIENT_FEATURE_SAVE_PATH)

    source_data_input_ = source_data.iloc[:, :source_data.shape[1] - 1]
    target_data_input_ = target_data.iloc[:, :target_data.shape[1] - 1]
    source_data_label_ = source_data.iloc[:, source_data.shape[1] - 1:]
    target_data_label_ = target_data.iloc[:, target_data.shape[1] - 1:]
    source_data_label_ = np.reshape(source_data_label_.values, [-1, 1])
    target_data_label_ = np.reshape(target_data_label_.values, [-1, 1])

    return source_data_input_, source_data_label_, target_data_input_, target_data_label_


def read_data_of_knowledge_feature(group):
    if group == 1:
        source_knowledge_feature = pd.read_excel(MIMIC_KNOWLEDGE_FEATURE_SAVE_PATH)
        target_knowledge_feature = pd.read_excel(PLAGH_KNOWLEDGE_FEATURE_SAVE_PATH)
    else:
        target_knowledge_feature = pd.read_excel(MIMIC_KNOWLEDGE_FEATURE_SAVE_PATH)
        source_knowledge_feature = pd.read_excel(PLAGH_KNOWLEDGE_FEATURE_SAVE_PATH)
    target_knowledge_feature = target_knowledge_feature.values
    source_knowledge_feature = source_knowledge_feature.values

    return source_knowledge_feature, target_knowledge_feature


def get_feature_embedding(group):
    if group == 1:
        source_type = 'mimic'
        source_data_input = pd.read_excel(MIMIC_PATIENT_FEATURE_SAVE_PATH)
    else:
        source_type = 'plagh'
        source_data_input = pd.read_excel(PLAGH_PATIENT_FEATURE_SAVE_PATH)
    # feature embedding one hot encode
    feature_embedding_one_hot = pd.DataFrame()
    feature_embedding_one_hot['造影'] = np.where(source_data_input['造影'] == 1, 1, 0)
    feature_embedding_one_hot['PCI'] = np.where(source_data_input['PCI'] == 1, 1, 0)
    feature_embedding_one_hot['抗血小板'] = np.where(source_data_input['抗血小板'] == 1, 1, 0)
    feature_embedding_one_hot['抗凝药物'] = np.where(source_data_input['抗凝药物'] == 1, 1, 0)
    feature_embedding_one_hot['beta受体拮抗剂'] = np.where(source_data_input['beta受体拮抗剂'] == 1, 1, 0)
    feature_embedding_one_hot['正性肌力药物'] = np.where(source_data_input['正性肌力药物'] == 1, 1, 0)
    feature_embedding_one_hot['血管扩张剂'] = np.where(source_data_input['血管扩张剂'] == 1, 1, 0)
    feature_embedding_one_hot['ACEI/ARB'] = np.where(source_data_input['ACEI/ARB'] == 1, 1, 0)
    feature_embedding_one_hot['钙通道阻滞剂'] = np.where(source_data_input['钙通道阻滞剂'] == 1, 1, 0)

    feature_embedding_one_hot['糖尿病'] = np.where(source_data_input['糖尿病'] == 1, 1, 0)
    feature_embedding_one_hot['瓣膜病'] = np.where(source_data_input['瓣膜病'] == 1, 1, 0)
    feature_embedding_one_hot['心肌病'] = np.where(source_data_input['心肌病'] == 1, 1, 0)
    feature_embedding_one_hot['冠心病'] = np.where(source_data_input['冠心病'] == 1, 1, 0)
    feature_embedding_one_hot['脑卒中'] = np.where(source_data_input['脑卒中'] == 1, 1, 0)
    feature_embedding_one_hot['心房颤动'] = np.where(source_data_input['心房颤动'] == 1, 1, 0)
    feature_embedding_one_hot['AHF急性心力衰竭'] = np.where(source_data_input['AHF急性心力衰竭'] == 1, 1, 0)

    feature_embedding_one_hot['女性'] = np.where(source_data_input['性别'] == 0, 1, 0)
    feature_embedding_one_hot['高钾血症'] = np.where(source_data_input['钾'] > 5.5, 1, 0)
    feature_embedding_one_hot['贫血'] = np.where(source_data_input['血红蛋白测定'] < 110, 1, 0)
    feature_embedding_one_hot['高龄'] = np.where(source_data_input['年龄'] <= 70, 0, 1)
    feature_embedding_one_hot['低血压'] = np.where((source_data_input['血压Low'] >= 60) |
                                                (source_data_input['血压high'] >= 90), 0, 1)
    feature_embedding_one_hot['高血压'] = np.where((source_data_input['血压Low'] >= 90) |
                                                (source_data_input['血压high'] >= 140), 1, 0)
    feature_embedding_one_hot['甘油三酯偏高'] = np.where(source_data_input['甘油三酯'] >= 2.26, 1, 0)
    feature_embedding_one_hot['丙氨酸氨基转移酶偏高'] = np.where(source_data_input['丙氨酸氨基转移酶'] > 40, 1, 0)
    feature_embedding_one_hot['总胆红素偏高'] = np.where(source_data_input['总胆红素'] > 21, 1, 0)
    feature_embedding_one_hot['高密度脂蛋白胆固醇偏低'] = np.where(source_data_input['高密度脂蛋白胆固醇'] >= 1, 0, 1)
    feature_embedding_one_hot['尿素氮偏高'] = np.where(source_data_input['尿素'] > 7.5, 1, 0)
    feature_embedding_one_hot['天冬氨酸氨基转移酶偏高'] = np.where(source_data_input['天冬氨酸氨基转移酶'] > 40, 1, 0)
    feature_embedding_one_hot['低密度脂蛋白胆固醇偏高'] = np.where(source_data_input['低密度脂蛋白胆固醇'] > 3.4, 1, 0)
    feature_embedding_one_hot['egfr偏低'] = np.where(source_data_input['egfr'] >= 80, 0, 1)
    feature_embedding_one_hot['脑利钠肽前体偏高'] = np.where(((source_data_input['年龄'] <= 50) &
                                                      (source_data_input['脑利钠肽前体'] >= 450)) | (
                                                                 (source_data_input['年龄'] <= 75) &
                                                                 (source_data_input['年龄'] > 50) &
                                                                 (source_data_input['脑利钠肽前体'] >= 900)) | (
                                                                 (source_data_input['年龄'] > 75) &
                                                                 (source_data_input['脑利钠肽前体'] >= 1800)), 1, 0)
    feature_embedding_one_hot.to_excel('feature_embedding_{}.xlsx'.format(source_type), index=False)
    return feature_embedding_one_hot


def get_knowledge_feature_extraction(group):
    
    # patient feature mapping to knowledge feature in one-hot
    feature_embedding_one_hot = get_feature_embedding(group)
  
    # train_flag=True: pre train the knowledge graph
    # train_flag=False: return the trained knowledge graph or error
    knowledge = knowledge_graph_pre_train(train_flag=True)
    feature_embedding_one_hot_copy = copy.deepcopy(feature_embedding_one_hot)
    for c in feature_embedding_one_hot_copy.columns:
        feature_embedding_one_hot_copy[c] = np.where(feature_embedding_one_hot_copy[c], c, 0)
    patient_feature_embedding_final = []
    for single_copy in feature_embedding_one_hot_copy.values:
        single_copy_filter = single_copy[np.where(single_copy != '0')]
        nodes_vector = knowledge.get_nodes_vector_by_source_node_only(single_copy_filter)
        patient_feature_embedding_final.append(nodes_vector.values)
    knowledge_feature_final = pd.DataFrame(patient_feature_embedding_final)
    if group == 1:
        knowledge_feature_final.to_excel(MIMIC_KNOWLEDGE_FEATURE_SAVE_PATH, index=False)
    else:
        knowledge_feature_final.to_excel(PLAGH_KNOWLEDGE_FEATURE_SAVE_PATH, index=False)

    return knowledge_feature_final.values


def knowledge_graph_pre_train(train_flag=True):
    return KnowledgeGraph(KNOWLEDGE_GRAPH_SOURCE_DATA_PATH, [50, 50], train_flag=train_flag)


if __name__ == "__main__":
    group = 1
    if group == 1:
        source_type = 'mimic'
        target_type = 'plagh'
    else:
        source_type = 'plagh'
        target_type = 'mimic'
    model_type = 'KAMA'

    # patient feature mapping to knowledge feature in one-hot
    mimic_feature_embedding_one_hot = get_feature_embedding(1)
    plagh_feature_embedding_one_hot = get_feature_embedding(2)
    
    # knowledge feature extraction
    mimic_patient_feature_embedding = get_knowledge_feature_extraction(1)
    plagh_patient_feature_embedding = get_knowledge_feature_extraction(2)

    # read patient feature
    source_data_input, source_data_label, target_data_input, target_data_label = read_data_of_sjz(group)
    MAX = source_data_input.max()
    MIN = source_data_input.min()
    source_data_input = (source_data_input-MIN)/(MAX-MIN)
    target_data_input = (target_data_input-MIN)/(MAX-MIN)
    source_data_input = source_data_input.values
    target_data_input = target_data_input.values

    # read knowledge_feature
    source_patient_feature_embedding, target_patient_feature_embedding = read_data_of_knowledge_feature()
    feature_embedding_input = source_patient_feature_embedding.shape[1]

    train_repeat = 5
    auc_list = []
    print(model_type)
    for i in range(train_repeat):
        print("iteration number: %d" % i)
        n_input = source_data_input.shape[1]
        k_folds = 5
        test_size = 1 / k_folds
        train_size = 1 - test_size
        split = StratifiedShuffleSplit(k_folds, test_size, train_size, 1).split(source_data_input, source_data_label)
        for ith_fold in range(k_folds):
            print('{} th fold of {} folds'.format(ith_fold, k_folds))
            train_index, test_index = next(split)
            source_train_sets = DataSet(source_data_input[train_index],
                                        source_data_label[train_index], source_patient_feature_embedding[train_index])
            source_test_sets = DataSet(source_data_input[test_index],
                                       source_data_label[test_index], source_patient_feature_embedding[test_index])
            target_data_sets = DataSet(target_data_input, target_data_label, target_patient_feature_embedding)
            tf.reset_default_graph()
            if model_type == 'MLP+AL':
                model = MCAL(n_input, feature_encoder_hidden=[30, 20, 10], n_class=1)
            elif model_type == 'KAMA':
                model = KAMA(n_input, feature_encoder_hidden=[30, 20], feature_embedding_input=feature_embedding_input,
                             n_class=1)
            elif model_type == 'KAMA_V1':
                model = KAMA_V1(n_input, feature_encoder_hidden=[30, 20],
                                feature_embedding_input=feature_embedding_input, n_class=1)
            elif model_type == 'KAMA_V2':
                model = KAMA_V2(n_input, feature_encoder_hidden=[30, 20],
                                feature_embedding_input=feature_embedding_input, n_class=1)
            elif model_type == 'KAMA_V3':
                model = KAMA_V3(n_input, feature_encoder_hidden=[30, 20],
                                feature_embedding_input=feature_embedding_input, n_class=1)
            elif model_type == 'KAMA_V4':
                model = KAMA_V4(n_input, feature_encoder_hidden=[30, 20],
                                feature_embedding_input=feature_embedding_input, n_class=1)
            elif model_type == 'MLP':
                model = MCMLP(n_input,  feature_encoder_hidden=[30, 20, 10], n_class=1)
            else:
                model = KAMA(n_input, feature_encoder_hidden=[30, 20], feature_embedding_input=feature_embedding_input,
                             n_class=1)

            result = model.fit(source_train_sets, source_test_sets, target_data_sets, target_data_sets, 256, keep_prob=0.8)
            auc_list.append(result)

    auc_list = pd.DataFrame(auc_list, columns=['source_test_auc', 'source_test_precision', 'source_test_recall',
                                               'source_test_f_score',
                                               'source_test_accuracy', 'target_test_auc', 'target_test_precision',
                                               'target_test_recall',
                                               'target_test_f_score', 'target_test_accuracy',
                                               'source_test_auc + target_test_auc'])
    auc_list.to_excel('result/auc_result_{}_{}_origin_data922_kg.xlsx'.format(source_type, model_type), index=False)


