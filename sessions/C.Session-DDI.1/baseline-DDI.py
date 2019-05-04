from tqdm import tqdm
from os import listdir
import pandas as pd
from xml.dom.minidom import parse
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

stopwords = set(stopwords.words('english'))

output_path_name = "task9.2_raquel_6.txt"

output_path = "evaluations/" + output_path_name
results_path = output_path.replace('.txt', '_All_scores.log')
datadir = '../../data/Test-DDI/DrugBank'
training_data = '/home/raquel/Documents/mai/ahlt/data/Train/All'
train_df_path = '/home/raquel/Documents/mai/ahlt/data/DF/train.csv'


def train_baseline():
    train_df = pd.read_csv(train_df_path, index_col=0)
    sentences_train = train_df.sentence_text.values
    y_train = train_df['relation_type'].values
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    print('training...')
    # classifier = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    print('trained')
    return vectorizer, classifier


vectorizer, classifier = train_baseline()


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
                    (is_ddi, ddi_type) = check_interaction(stext)
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
