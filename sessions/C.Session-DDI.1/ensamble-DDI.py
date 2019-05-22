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
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression

from utils import get_entity_dict, smaller_subtree_containing_the_drugs, preprocess

stopwords = set(stopwords.words('english'))

output_path_name = "task9.2_cascade_rf_mlp_99.txt"

output_path = "evaluations/" + output_path_name
results_path = output_path.replace('.txt', '_All_scores.log')
datadir = '../../data/Test-DDI/DrugBank'
training_data = '/home/raquel/Documents/mai/ahlt/data/Train/All'
train_df_path = '../../data/DF/train.csv'
processed_train_df_path = '../../data/DF/train_processed.csv'

encoder = LabelBinarizer()
tokenizer = Tokenizer()
vectorizer = CountVectorizer()


def tokenize_and_padd(sentences_train):
    X_train = tokenizer.texts_to_sequences(sentences_train)
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    max_s = [len(x) for x in X_train]
    maxlen = np.max(max_s)
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

    return X_train, vocab_size, maxlen


def train_baseline():
    train_df = pd.read_csv(train_df_path, index_col=0)

    sentences_train, dictionary, y_train_encoded = preprocess(train_df, processed_train_df_path, encoder)

    y_train = train_df['relation_type'].values

    y_binary = ['none' if i == 'none' else 'interaction'for i in y_train]

    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    # tokenizer.fit_on_texts(sentences_train)
    #
    # X_train, vocab_size, maxlen = tokenize_and_padd(sentences_train)

    print('training...')

    binary_classifier = RandomForestClassifier(n_estimators=500, max_depth=90, n_jobs=-1,
                                               class_weight='balanced')

    binary_classifier.fit(X_train, y_binary)

    mlp = MLPClassifier(activation='tanh', alpha=0.1, hidden_layer_sizes=(30, 5), learning_rate='constant')

    # classifier = VotingClassifier(estimators=[('Random Forest', rf), ('MLP', mlp)],
    #                             voting='soft')
    classifier = mlp
    classifier.fit(X_train[np.array(y_binary)=='interaction',:], y_train[np.array(y_binary)=='interaction'])
    print('trained')
    return binary_classifier, classifier, dictionary


binary_classifier, classifier, dictionary = train_baseline()


def check_interaction(sentence):
    # uses the vectorizer and the classifier already trained
    sentence_array = vectorizer.transform([sentence])
    # sentence_array = tokenizer.texts_to_sequences([sentence])
    # sentence_array = pad_sequences(sentence_array, padding='post', maxlen=maxlen)

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
