from tqdm import tqdm
from os import listdir
import pandas as pd
from xml.dom.minidom import parse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from utils import get_entity_dict, smaller_subtree_containing_the_drugs, preprocess
from sklearn.preprocessing import LabelBinarizer


output_path_name = "task9.2_mlp_final_42.txt"

output_path = "evaluations/" + output_path_name
results_path = output_path.replace('.txt', '_All_scores.log')
datadir = '../../data/Test-DDI/DrugBank'
train_df_path = '../../data/DF/train.csv'
processed_train_df_path = '../../data/DF/train_processed.csv'

vectorizer = CountVectorizer()
encoder = LabelBinarizer()


def train_baseline():
    train_df = pd.read_csv(train_df_path, index_col=0)

    sentences_train, dictionary, y_train_encoded = preprocess(train_df, processed_train_df_path, encoder)
    y_train = train_df['relation_type'].values
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    print('training...')
    classifier = MLPClassifier(activation='tanh', alpha= 0.1, hidden_layer_sizes=(30, 5), learning_rate='constant')
    classifier.fit(X_train, y_train)
    print('trained')
    return classifier, dictionary


classifier,dictionary = train_baseline()


def check_interaction(sentence):
    # uses the vectorizer and the classifier already trained
    sentence_array = vectorizer.transform([sentence])
    y_pred = classifier.predict(sentence_array)

    if y_pred[0] == 'none':
        return False, "null"
    else:
        return True, y_pred[0]


def predict(datadir, output_path, test=False):
    # process each file in directory
    with open(output_path, 'w') as file:
        for f in tqdm(listdir(datadir)):

            # parse XML file, obtaining a DOM tree
            tree = parse(datadir + "/" + f)

            # process each sentence in the file
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                entity_dict = get_entity_dict(s)
                sid = s.attributes["id"].value  # get sentence id
                stext = s.attributes["text"].value  # get sentence text

                # load sentence entities
                entities = {}
                ents = s.getElementsByTagName("entity")
                for e in ents:
                    id = e.attributes["id"].value
                    offs = e.attributes["charOffset"].value.split("-")
                    entities[id] = offs

                # for each pair in the sentence, decide whether it is DDI and its type
                pairs = s.getElementsByTagName("pair")
                for p in pairs:
                    id_e1 = p.attributes["e1"].value
                    id_e2 = p.attributes["e2"].value

                    e1 = entity_dict[id_e1]
                    e2 = entity_dict[id_e2]

                    try:
                        ddi_type = dictionary[e1][e2]
                        is_ddi = ddi_type != 'null'
                        # print(ddi_type)
                    except KeyError:
                        processed_sentence = smaller_subtree_containing_the_drugs(stext, [e1, e2])
                        (is_ddi, ddi_type) = check_interaction(processed_sentence)
                    ddi = "1" if is_ddi else "0"
                    file.write(sid + "|" + id_e1 + "|" + id_e2 + "|" + ddi + "|" + ddi_type)
                    file.write('\n')
                    if test:
                        return


def show_results(results_path):
    import subprocess
    subprocess.call(['java', '-jar', '../../eval/evaluateDDI.jar', '../../data/Test-DDI/All', output_path])
    results_file = open(results_path, 'r')
    print(results_file.read())
    results_file.close()


predict(datadir, output_path, False)
show_results(results_path)
