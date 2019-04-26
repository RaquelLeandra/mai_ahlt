# LSTM-CRF for sequence tagging using keras
# reference: https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/
import pandas as pd
import numpy as np
import pickle

# read csv data set and transform it in order to feed to the network
# example:
# input -> [The, benefits, of, waxanoryl, should, be, ...]
# output -> [O, O, O, B-drug, O, O, ...]
def get_preprocessed_data(train_csv_path, test_csv_path):

    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)
    df = pd.concat([df_train, df_test])

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
    # 0 is reserved for padding
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}

    from keras.preprocessing.sequence import pad_sequences
    x = [[word2idx[w] for w in s] for s in x]
    # TODO: Probably we don't need such a long max_length just for a few sentences (101)
    max_length = max([len(s) for s in x])
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
x_train = d['x_train']
y_train = d['y_train']
x_test = d['x_test']
y_test = d['y_test']
max_length = d['max_length']
n_words = d['n_words']
n_tags = d['n_tags']
tag2idx = d['tag2idx']


# train the model
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

input = Input(shape=(max_length,))
model = Embedding(input_dim=n_words + 1, output_dim=20,
                  input_length=max_length, mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
print(model.summary())

history = model.fit(x_train, np.array(y_train), batch_size=32, epochs=10,
                    validation_split=0.1, verbose=1)

# predict the name entities in the test set
# evaluate the model
from sklearn.metrics import classification_report
y_pred = model.predict(x_test, verbose=1)

idx2tag = {i: w for w, i in tag2idx.items()}

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i])
        out.append(out_i)
    return out

pred_labels = pred2label(y_pred)
test_labels = pred2label(y_test)


flattened_pred_labels = np.array([x for y in pred_labels for x in y])
flattened_test_labels = np.array([x for y in test_labels for x in y])
# filter O tags
flattened_pred_labels = flattened_pred_labels[np.argwhere(flattened_test_labels != 'O')]
flattened_test_labels = flattened_test_labels[np.argwhere(flattened_test_labels != 'O')]

print(classification_report(flattened_pred_labels, flattened_test_labels))


# transform the predictions into a file in the evaluation format
# example: DDI-MedLine.d203.s1|129-143|antidepressants|group
# iterate through sentences of test data set and for each one look at the predictions.
# when one named entity is found, look at the char offset in the data frame, and
# construct the line sentence_id|char_offset|text|entity_type








# run evaulation script with "subprocess"
