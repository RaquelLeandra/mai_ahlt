# LSTM-CRF for sequence tagging using keras
# reference: https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/
import pandas as pd
import numpy as np
import pickle
import os

# read csv data set and transform it in order to feed to the network
# example:
# input -> [The, benefits, of, waxanoryl, should, be, ...]
# output -> [O, O, O, B-drug, O, O, ...]
def get_preprocessed_data(train_csv_path, test_csv_path):

    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)
    df = pd.concat([df_train, df_test])
    df['tag'].values[pd.isnull(df['tag'].values)] = 'O'

    x = []
    y = []
    # unique() respects the original order of the data frame
    for id in df.sentence_id.unique():
        words = list(df[df['sentence_id'] == id]['word'].values)
        tags = list(df[df['sentence_id'] == id]['tag'].values)
        x.append(words)
        y.append(tags)


    words = list(set(df["word"].values))
    tags = list(set(df["tag"].values))
    n_words = len(words)
    n_tags = len(tags)

    # max_length = max([len(s) for s in x])
    # ALTERNATIVE: MAX LENGTH TO REASONABLE VALUE AND CROP IF SENTENCE IS LONGER
    max_length = 50
    x = [s[:max_length] for s in x]
    y = [s[:max_length] for s in y]
    # 0 is reserved for padding
    word2idx = {w: i for i, w in enumerate(words)}
    # NaN issue related with "]" at the end of the word and other unkown issue
    tag2idx = {t: i for i, t in enumerate(tags)}

    from keras.preprocessing.sequence import pad_sequences
    x = [[word2idx[w] for w in s] for s in x]
    # TODO: Probably we don't need such a long max_length just for a few sentences (101)
    x = pad_sequences(maxlen=max_length, sequences=x, padding="post", value=0)

    y = [[tag2idx[t] for t in s] for s in y]
    # we need to assign "O" to every padding element
    y = pad_sequences(maxlen=max_length, sequences=y, padding="post", value=tag2idx['O'])
    from keras.utils import to_categorical
    y = [to_categorical(i, num_classes=n_tags) for i in y]

    # separate train and test sets again
    n_test_sentences = len(df_test.sentence_id.unique())
    x_train = x[:-n_test_sentences]
    y_train = y[:-n_test_sentences]
    x_test = x[-n_test_sentences:]
    y_test = y[-n_test_sentences:]

    d = {}
    d['x_train'] = x_train
    d['y_train'] = y_train
    d['x_test'] = x_test
    d['y_test'] = y_test
    d['max_length'] = max_length
    d['n_words'] = n_words
    d['n_tags'] = n_tags
    d['tag2idx'] = tag2idx
    d['word_index'] = word2idx

    return d

train_csv_path = 'train_set.csv'
test_csv_path = 'test_set.csv'
UPDATE_DATA = False

if UPDATE_DATA:
    data = get_preprocessed_data(train_csv_path, test_csv_path)
    f = open('data.pkl', 'wb')
    pickle.dump(data, f)
    f.close()

f = open('data.pkl', 'rb')
d = pickle.load(f)
f.close()
x_train = d['x_train']
y_train = d['y_train']
x_test = d['x_test']
y_test = d['y_test']
max_length = d['max_length']
n_words = d['n_words']
n_tags = d['n_tags']
tag2idx = d['tag2idx']
word_index = d['word_index']

# build pretraining embedding matrix
embedding_path = 'glove.6B.300d.txt'
embedding_dim = 300
embeddings_index = {}
f = open(embedding_path, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    if word in word_index:
        word_coefs = np.asarray(values[1:])
        embeddings_index[word] = word_coefs
f.close()

# map word indices from Tokenizer to embedding coefficients
embedding_matrix = np.zeros((n_words,embedding_dim))

for word, index in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


f = open('embedding_matrix.pkl','wb')                                  
pickle.dump(embedding_matrix, f)
f.close()
