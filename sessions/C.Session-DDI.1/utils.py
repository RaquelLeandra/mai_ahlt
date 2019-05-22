from nltk.stem import WordNetLemmatizer
from nltk.tag import PerceptronTagger
from nltk.corpus import wordnet as wn
from nltk.tree import *
import re
from tqdm import tqdm
from os import listdir
import pandas as pd

from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

to_remove = ['-LRB-', '-RRB-', ',', '.', 'a', 'A']


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
                clean.append(w.lower())
    return clean


def all_drugs_in_tree(target_drugs, leafs):
    for drug in target_drugs:
        if re.sub(r'[^\w]', '', drug) not in leafs:
            return False
    return True


def smaller_subtree_containing_the_drugs(sentence, target_drugs):
    tree_string = nlp.annotate(sentence, properties={'annotators': 'parse', 'outputFormat': 'json'})
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


def show_results(results_path):
    import subprocess
    subprocess.call(['java', '-jar', '../../eval/evaluateDDI.jar', '../../data/Test-DDI/All', output_path])
    results_file = open(results_path, 'r')
    print(results_file.read())
    results_file.close()
