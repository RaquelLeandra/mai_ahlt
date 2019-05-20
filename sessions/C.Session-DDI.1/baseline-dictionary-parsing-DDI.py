from tqdm import tqdm
from os import listdir
import pandas as pd
from xml.dom.minidom import parse
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from nltk.tag import PerceptronTagger
from nltk.corpus import wordnet as wn
from nltk.tree import *
import re

from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

stopwords = set(stopwords.words('english'))

to_remove = ['-LRB-', '-RRB-', ',', '.', 'a', 'A']
output_path_name = "task9.2_raquel_6.txt"

output_path = "evaluations/" + output_path_name
results_path = output_path.replace('.txt', '_All_scores.log')
datadir = '../../data/Test-DDI/DrugBank'
training_data = '../../data/Train/All'
train_df_path = '../../data/DF/train.csv'


def get_entity_dict(sentence_dom):
    entities = sentence_dom.getElementsByTagName('entity')
    entity_dict = {}
    for entity in entities:
        id = entity.getAttribute('id')
        word = entity.getAttribute('text')
        entity_dict[id] = word
    return entity_dict


def penn_to_wn(tag):
    if tag in ['JJ', 'JJR', 'JJS']:
        return wn.ADJ
    elif tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        return wn.NOUN
    elif tag in ['RB', 'RBR', 'RBS']:
        return wn.ADV
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return wn.VERB
    return wn.NOUN


def preprocessor_lemmatize(tagged):
    lemmatizer = WordNetLemmatizer()
    result = []
    for word, tag in tagged:
        # A verb for sure if it ends in ing (?).
        # Sometimes, verbs are wrongly classified, for example, "A man is smoking" (Smoking => Noun)
        lemma = word
        if not 'NN' in tag:
            if word.endswith('ing'):
                tag = 'VB'
            lemma = lemmatizer.lemmatize(word, penn_to_wn(tag))
        result.append((lemma, tag))
    return result


def clean_sentence(sentence_splitted):
    clean = []
    for w in sentence_splitted:
        if w not in to_remove:
            w = re.sub(r'([^a-zA-Z\s]+?)', '', w)
            if w:
                clean.append(w)
    return clean


def all_drugs_in_tree(target_drugs, leafs):
    for drug in target_drugs:
        if re.sub(r'[^\w]', '', drug) not in leafs:
            return False
    return True


def smaller_subtree_containing_the_drugs(sentence, target_drugs):
    tree_string = nlp.annotate(sentence, properties={'annotators': 'parse', 'outputFormat': 'json'})
    if len(tree_string['sentences']) > 1:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', len(tree_string['sentences']))
    tagger = PerceptronTagger()
    best_subtree = None
    size = 9999999
    target_drugs = [dr for drug in target_drugs for dr in drug.split(' ')]
    for s in tree_string['sentences']:
        tree_parsed = Tree.fromstring(s['parse'])
        for subtree in tree_parsed.subtrees():
            #         print(subtree.pretty_print())
            leafs = subtree.leaves()
            current_size = len(leafs)
            if all_drugs_in_tree(target_drugs, leafs):
                if current_size < size:
                    best_subtree = subtree
                    size = current_size
        #                 print(subtree.leaves())

    try:
        clean = clean_sentence(best_subtree.leaves())
    except:
        clean = clean_sentence(sentence.split())
    # print('clean',clean)
    tagged = tagger.tag(clean)
    # print('tag:', tagged)
    lemmatized = preprocessor_lemmatize(tagged)
    # print('lemmatized', lemmatized)
    new_sentence = ' '.join([l for l, t in lemmatized])

    return new_sentence


def train_baseline():
    train_df = pd.read_csv(train_df_path, index_col=0)

    for index, row in train_df.iterrows():
        print(train_df.loc[index, 'sentence_text'], train_df.loc[index, ['e1', 'e2']])
        new_sentence = smaller_subtree_containing_the_drugs(train_df.loc[index, 'sentence_text'],
                                                            train_df.loc[index, ['e1', 'e2']])
        train_df.loc[index, 'sentence_text'] = new_sentence
    train_df.to_csv('new_train.csv')

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
