import os
from xml.dom import minidom
import pandas as pd


def get_entity_dict(sentence_dom):
    entities = sentence_dom.getElementsByTagName('entity')
    entity_dict = {}
    for entity in entities:
        id = entity.getAttribute('id')
        word = entity.getAttribute('text')
        entity_dict[id] = word
    return entity_dict


def get_dataframe_from_data(dataset_csv_file, directory=None):
    types = set()
    parsed_data = []
    total_files_to_read = os.listdir(directory)
    print('Total files to read:', len(total_files_to_read), ' from dir: ', directory)
    for file in total_files_to_read:
        file = directory + '/' + file

        DOMTree = minidom.parse(file)
        sentences = DOMTree.getElementsByTagName('sentence')

        for sentence_dom in sentences:

            entity_dict = get_entity_dict(sentence_dom)
            pairs = sentence_dom.getElementsByTagName('pair')
            sentence_text = sentence_dom.getAttribute('text')
            sid = sentence_dom.getAttribute('id')
            for pair in pairs:
                ddi_flag = pair.getAttribute('ddi')

                if ddi_flag == 'true':
                    relation_type = pair.getAttribute('type')
                else:
                    relation_type = 'none'

                e1 = pair.getAttribute('e1')
                e2 = pair.getAttribute('e2')
                if relation_type != '':
                    types.add(relation_type)
                    parsed_data.append([entity_dict[e1], entity_dict[e2], relation_type, sentence_text, sid, e1, e2])

    print(types)
    df = pd.DataFrame(parsed_data, columns='e1,e2,relation_type,sentence_text,sid,id_e1,id_e2'.split(','))
    df.to_csv(dataset_csv_file)
    df = pd.read_csv(dataset_csv_file, index_col=0)
    return df


test = '/home/raquel/Documents/mai/ahlt/data/Test-DDI/All'
train = '/home/raquel/Documents/mai/ahlt/data/Train/All'
test_df_path = '/home/raquel/Documents/mai/ahlt/data/DF/test.csv'
train_df_path = '/home/raquel/Documents/mai/ahlt/data/DF/train.csv'

df_train = get_dataframe_from_data(train_df_path, train)
df_test = get_dataframe_from_data(test_df_path, test)
