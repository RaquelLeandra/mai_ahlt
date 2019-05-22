from tqdm import tqdm
from os import listdir
import pandas as pd
from xml.dom.minidom import parse
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelBinarizer

from keras.layers import Dense, Input, Flatten, Reshape, concatenate, Dropout
from keras.layers import  Conv2D, MaxPooling2D, Embedding
from keras.models import Model
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from utils import get_entity_dict, smaller_subtree_containing_the_drugs, preprocess

stopwords = set(stopwords.words('english'))

output_path_name = "task9.2__cascade_cnn_70.txt"

output_path = "evaluations/" + output_path_name
results_path = output_path.replace('.txt', '_All_scores.log')
datadir = '../../data/Test-DDI/DrugBank'
training_data = '/home/raquel/Documents/mai/ahlt/data/Train/All'
train_df_path = '../../data/DF/train.csv'
processed_train_df_path = '../../data/DF/train_processed.csv'

encoder = LabelBinarizer()
encoder_bin = LabelBinarizer()

tokenizer = Tokenizer()
vectorizer = CountVectorizer()

def kimCNN(embedding_output_size, imput_size, vocab_size, num_labels=5,loss='categorical_crossentropy'):
    """
    Convolution neural network model for sentence classification.
    Parameters
    ----------
    embedding_output_size: Dimension of the embedding space.
    vocab_size: size of the vocabulary
    imput_size: number of features of the imput.
    num_labels: number of output labels
    Returns
    -------
    compiled keras model
    """
    print('Preparing embedding matrix.')

    embedding_layer = Embedding(input_dim=vocab_size,
                                output_dim=embedding_output_size,
                                input_length=imput_size,
                                trainable=True)

    print('Training model.')

    sequence_input = Input(shape=(imput_size,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    print(embedded_sequences.shape)

    # add first conv filter
    embedded_sequences = Reshape((imput_size, embedding_output_size, 1))(embedded_sequences)
    x = Conv2D(100, (5, embedding_output_size), activation='relu')(embedded_sequences)
    x = MaxPooling2D((imput_size - 5 + 1, 1))(x)

    # add second conv filter.
    y = Conv2D(100, (4, embedding_output_size), activation='relu')(embedded_sequences)
    y = MaxPooling2D((imput_size - 4 + 1, 1))(y)

    # add third conv filter.
    z = Conv2D(100, (3, embedding_output_size), activation='relu')(embedded_sequences)
    z = MaxPooling2D((imput_size - 3 + 1, 1))(z)

    # concate the conv layers
    alpha = concatenate([x, y, z])

    # flatted the pooled features.
    alpha = Flatten()(alpha)

    # dropout
    alpha = Dropout(0.5)(alpha)

    # predictions
    preds = Dense(num_labels, activation='softmax')(alpha)

    # build model
    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta()

    model.compile(loss=loss,
                  optimizer=adadelta,
                  metrics=['acc'])
    model.summary()

    return model


def train_cnn():
    train_df = pd.read_csv(train_df_path, index_col=0)
    sentences_train, dictionary, y_train = preprocess(train_df, processed_train_df_path, encoder,cascade=True)

    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    max_s = [len(x) for x in X_train]
    maxlen = np.max(max_s)
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

    word_embedding_size = 200

    # Binary
    y_binary = ['none' if i == 'none' else 'interaction' for i in y_train]
    y_bin_encoded = encoder_bin.fit_transform(y_binary)

    y_train_encoded = encoder.fit_transform(y_train[np.array(y_binary)=='interaction'])

    binary_classifier = kimCNN(embedding_output_size=word_embedding_size, imput_size=X_train.shape[1],
                               vocab_size=vocab_size,
                               num_labels=1,loss='binary_crossentropy')

    binary_classifier.fit(X_train, y_bin_encoded,
                          epochs=30,
                          verbose=True,
                          batch_size=100,
                          validation_split=0.1,
                          class_weight='auto')

    classifier = kimCNN(embedding_output_size=word_embedding_size, imput_size=X_train.shape[1], vocab_size=vocab_size,
                        num_labels=4)

    classifier.fit(X_train[np.array(y_binary)=='interaction',:], y_train_encoded,
                   epochs=30,
                   verbose=True,
                   batch_size=100,
                   validation_split=0.1,
                   class_weight='auto')

    print('trained')
    return binary_classifier, classifier, dictionary, maxlen


binary_classifier, classifier, dictionary, maxlen = train_cnn()


def check_interaction(sentence):
    # uses the vectorizer and the classifier already trained
    sentence_array = tokenizer.texts_to_sequences([sentence])
    sentence_array = pad_sequences(sentence_array, padding='post', maxlen=maxlen)
    # print(sentence_array)
    y_bin = binary_classifier.predict(sentence_array)
    if y_bin[0] > 0.6:
        return False, "null"
    else:
        y_probs = classifier.predict(sentence_array)
        y_class = np.argmax(y_probs, axis=1)
        y_pred = encoder.classes_[y_class]
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
