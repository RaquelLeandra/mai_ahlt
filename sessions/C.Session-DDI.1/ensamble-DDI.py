from tqdm import tqdm
from os import listdir
import pandas as pd
from xml.dom.minidom import parse
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

from utils import get_entity_dict, smaller_subtree_containing_the_drugs

stopwords = set(stopwords.words('english'))

output_path_name = "task9.2_ensamble_cascade_90.txt"

output_path = "evaluations/" + output_path_name
results_path = output_path.replace('.txt', '_All_scores.log')
datadir = '../../data/Test-DDI/DrugBank'
training_data = '/home/raquel/Documents/mai/ahlt/data/Train/All'
train_df_path = '/home/raquel/Documents/mai/ahlt/data/DF/train.csv'


def train_baseline():
    train_df = pd.read_csv('saved_train_nice.csv', index_col=0)

    # for index, row in train_df.iterrows():
    #     # print(train_df.loc[index, 'sentence_text'], train_df.loc[index, ['e1', 'e2']])
    #     new_sentence = smaller_subtree_containing_the_drugs(train_df.loc[index, 'sentence_text'],
    #                                                         train_df.loc[index, ['e1', 'e2']])
    #     train_df.loc[index, 'sentence_text'] = new_sentence
    # train_df.to_csv('saved_train_nice.csv')
    dictionary = {}
    for index, row in train_df.iterrows():
        d_1 = row['e1'].lower()
        d_2 = row['e2'].lower()
        interaction = row['relation_type']
        if interaction == 'none':
            interaction = 'null'
        if d_1 not in dictionary:
            dictionary[d_1] = {}
        if d_2 not in dictionary:
            dictionary[d_2] = {}
        dictionary[d_1][d_2] = interaction
        dictionary[d_2][d_1] = interaction

    sentences_train = train_df.sentence_text.values
    y_train = train_df['relation_type'].values

    y_binary = ['none' if i == 'none' else 'interaction'for i in y_train]
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    print('training...')

    binary_classifier = RandomForestClassifier(n_estimators=500, max_depth=90, n_jobs=-1,
                                class_weight='balanced', random_state=0)

    binary_classifier.fit(X_train, y_binary)

    rf = RandomForestClassifier(n_estimators=500, max_depth=60, n_jobs=-1,
                                class_weight='balanced')

    mlp = MLPClassifier(activation='tanh', alpha=0.1, hidden_layer_sizes=(30, 5), learning_rate='constant')

    classifier = VotingClassifier(estimators=[('Random Forest', rf), ('MLP', mlp)],
                                voting='soft')
    classifier.fit(X_train[np.array(y_binary)=='interaction',:], y_train[np.array(y_binary)=='interaction'])
    print('trained')
    return vectorizer, binary_classifier, classifier, dictionary


vectorizer, binary_classifier, classifier, dictionary = train_baseline()


def check_interaction(sentence):
    # uses the vectorizer and the classifier already trained
    sentence_array = vectorizer.transform([sentence])
    y_bin = binary_classifier.predict(sentence_array)

    if y_bin[0] == 'none':
        return False, "null"
    else:
        y_pred = classifier.predict(sentence_array)
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
