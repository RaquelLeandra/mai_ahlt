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
    word2idx = {w: i + 1 for i, w in enumerate(words)}
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

# load embedding_matrix
f = open('embedding_matrix.pkl','rb')
embedding_matrix = pickle.load(f)
embedding_dim = 300

# train the model
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras.initializers import Constant

input = Input(shape=(max_length,))
model = Embedding(n_words, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                input_length=max_length, trainable=False)(input)  # 50-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.2))(model)  # variational biLSTM
model = Bidirectional(LSTM(units=50, return_sequences=True,
                            recurrent_dropout=0.2))(model)
model = TimeDistributed(Dense(100, activation="relu"))(model)  # a dense layer as suggested by neuralNer
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
print(model.summary())

history = model.fit(x_train, np.array(y_train), batch_size=64, epochs=40,
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
df_test = pd.read_csv(test_csv_path)
df_test['sentence_ix'] = pd.factorize(df_test['sentence_id'])[0]
sentence_lengths = df_test.groupby(df_test['sentence_ix'])['word_ix'].max().add(1).values

unpadded_pred_labels = [pred_labels[i][:sentence_lengths[i]] for i in range(len(pred_labels))]

def get_tag(tag):
    # UNKNOWN BUG: Nan??
    if type(tag) == float:
        print('One NaN found!')
        return 'drug'
    return tag.split('-')[1]

def write_entity_to_file(s_ix, start_word_ix, end_word_ix, current_tag):
    df_entity = df_test[df_test['sentence_ix'] == s_ix]\
                       [df_test['word_ix'].isin(range(start_word_ix, end_word_ix+1))]
    sentence_id = df_entity.iloc[0].sentence_id
    offset_start = df_entity.iloc[0].offset_start
    offset_end = df_entity.iloc[-1].offset_end
    words = ' '.join(df_entity.word.values)
    tag = current_tag
    with open('testset_results.txt','a') as f:
        f.write('{}|{}-{}|{}|{}\n'.format(sentence_id, offset_start,
                offset_end, words, tag))

# remove test result file
os.remove('testset_results.txt')

for s_ix in range(len(unpadded_pred_labels)):
    s = unpadded_pred_labels[s_ix]
    start_word_ix = None
    end_word_ix = None
    current_tag = None
    for w_ix in range(len(s)):
        # CASE 1: FOUND ENTITY TAG AND NO CURRENT ENTITY ACTIVE
        # ACTION: INITIALIZE NEW ENTITY
        if (start_word_ix is None and s[w_ix] != 'O'):
            # initialize current entity
            start_word_ix = w_ix
            end_word_ix = w_ix
            current_tag = get_tag(s[w_ix])
        # CASE 2: THERE IS AN ACTIVE ENTITY BUT CURRENT TAG IS 'O'
        # ACTION: END ENTITY SEARCH, WRITE IT TO FILE, RESET ENTITY VARIABLES
        elif (s[w_ix] == 'O' and start_word_ix is not None):
            #write row
            write_entity_to_file(s_ix, start_word_ix, end_word_ix, current_tag)
            #reset variables
            start_word_ix = None
            end_word_ix = None
            current_tag = None
        # CASE 3: THERE IS AN ACTIVE ENTITY AND NEXT TAG IS A CONTINUATION
        # ACTION: ENLARGE/UPDATE CURRENT ACTIVE ENTITY
        elif (current_tag is not None and current_tag == get_tag(s[w_ix]) and s[w_ix][0] == 'I'):
            # update current entity info
            end_word_ix = w_ix
        # CASE 4: THERE IS AN ACTIVE ENTITY BUT NEXT TAG IS NOT A CONTINUATION
        # ACTION: WRITE CURRENT ACTIVE ENTITY TO FILE, START A NEW ACTIVE ENTITY
        elif (current_tag is not None):
           #write past row to file
           write_entity_to_file(s_ix, start_word_ix, end_word_ix, current_tag)
           #start new row info
           start_word_ix = w_ix
           end_word_ix = w_ix
           current_tag = get_tag(s[w_ix])

os.system('java -jar ../../eval/evaluateNER.jar ../../data/Test-NER/All testset_results.txt')



f = open('testset_results.txt', 'w')

for _, row in df_test.iterrows():
    if row['word_ix'] >= max_length: continue
    tag = pred_labels[row['sentence_ix']][row['word_ix']]
    if tag != 'O':
        f.write('{}|{}-{}|{}|{}\n'.format(row['sentence_id'], row['offset_start'],
                row['offset_end'], row['word'], tag.split('-')[1]))

f.close()

# print results
os.system('java -jar ../../eval/evaluateNER.jar ../../data/Test-NER/All testset_results.txt')







# run evaulation script with "subprocess"
