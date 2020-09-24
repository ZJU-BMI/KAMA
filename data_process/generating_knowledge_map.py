from experiment import *


def generate_GK():
    node_dict_map = dict()
    # relation_dict_list = []
    kg = []
    whole_data = pd.read_excel(KNOWLEDGE_GRAPH_SOURCE_DATA_PATH, sheet_name=KNOWLEDGE_GRAPH_DATA_SHEET)
    node_dict_list = whole_data[KNOWLEDGE_GRAPH_DATA_SHEET[0]].values
    relation_dict_list = whole_data[KNOWLEDGE_GRAPH_DATA_SHEET[1]].values
    for node in node_dict_list:
        kg.append(
            'CREATE (:%s {chineseName: "%s",englishName:"%s",abbreviation:"%s"});' % (
            node[3], node[0], node[1], node[2]))
        node_dict_map[node[0]] = node[3]

    for relation in relation_dict_list:
        try:
            kg.append('MATCH (u:%s{chineseName:"%s"}), (r:%s{chineseName:"%s"}) CREATE (u)-[:%s]->(r);' %
                      (node_dict_map[relation[0]], relation[0], node_dict_map[relation[2]], relation[2], relation[1]))
        except:
            pass
    result = '\n'.join(kg)
    with open('J:/graph9223.txt', 'a') as f:
        f.write(result)


if __name__ == '__main__':
    # main()
    generate_GK()
