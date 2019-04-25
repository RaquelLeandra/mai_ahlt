#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd


# -------- classify_token ----------
# -- check if a token is a drug, and of which type

def classify_token(txt):
    # Complete this function to return a pair (boolean, drug_type)
    # depending on whether the token is a drug name or not
    return False, ""


# --------- tokenize sentence -----------
# -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    # word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        # keep track of the position where each token should appear, and
        # store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset + len(t) - 1))
        offset += len(t)

    # tks is a list of triples (word,start,end)
    return tks


# --------- get tag -----------
#  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans):
    (form, start, end) = token
    for (spanS, spanE, spanT) in spans:
        if start == spanS and end <= spanE:
            return "B-" + spanT
        elif start >= spanS and end <= spanE:
            return "I-" + spanT

    return "O"


# --------- Feature extractor -----------
# -- Extract features for each token in given sentence

def extract_features(tokens, pos_tags, info_lists):
    # for each token, generate list of features and add it to the result
    result = []
    for k in range(0, len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]


        tokenFeatures.append("form=" + t)
        tokenFeatures.append("formlower=" + t.lower())
        tokenFeatures.append("suf3=" + t[-3:])
        tokenFeatures.append("suf4=" + t[-4:])
        tokenFeatures.append("size=" + str(len(t)))
        if len(t) <=3:
            tokenFeatures.append("isSmall")
        if t.isupper():
            tokenFeatures.append("isUpper")
        elif t[0].isupper():
            tokenFeatures.append("isCapital")
        if t.istitle():
            tokenFeatures.append("isTitle")
        if t.isdigit():
            tokenFeatures.append("isDigit")
        if '-' in t:
            tokenFeatures.append("hasDash")
        if t.lower() in info_lists['drug_list']:
            tokenFeatures.append('inDrugList')
        #if t.lower() in info_lists['common_suffixes']:
        #    tokenFeatures.append('commonSuffix')

        if k > 0:
            tPrev = tokens[k - 1][0]
            tokenFeatures.append("formPrev=" + tPrev)
            tokenFeatures.append("formlowerPrev=" + tPrev.lower())
            tokenFeatures.append("suf3Prev=" + tPrev[-3:])
            tokenFeatures.append("suf4Prev=" + tPrev[-4:])
            tokenFeatures.append("postagPrev=" + pos_tags[k - 1][1])
            #if (tPrev.isupper()): tokenFeatures.append("isUpperPrev")
            #if (tPrev.istitle()): tokenFeatures.append("isTitlePrev")
            #if (tPrev.isdigit()): tokenFeatures.append("isDigitPrev")
        else:
            tokenFeatures.append("BoS")

        if k < len(tokens) - 1:
            tNext = tokens[k + 1][0]
            tokenFeatures.append("formNext=" + tNext)
            tokenFeatures.append("formlowerNext=" + tNext.lower())
            tokenFeatures.append("suf3Next=" + tNext[-3:])
            tokenFeatures.append("suf4Next=" + tNext[-4:])
            tokenFeatures.append("postagNext=" + pos_tags[k + 1][1])
            #if (tNext.isupper()): tokenFeatures.append("isUpperNext")
            #if (tNext.istitle()): tokenFeatures.append("isTitleNext")
            #if (tNext.isdigit()): tokenFeatures.append("isDigitNext")
        else:
            tokenFeatures.append("EoS")

        result.append(tokenFeatures)

    return result

from collections import Counter
def get_info_lists():
    d = {}
    # list of drug names
    path = 'drug_FDA_database.csv'
    df = pd.read_csv(path, sep=';')
    drug_list = df.DrugName.apply(str.lower).values
    d['drug_list'] = set(drug_list)
    # list of common 4-char suffixes
    suffixes = [x[-4:] for x in drug_list]
    common_suffixes = [x[0] for x in Counter(suffixes).most_common()[:100]]
    d['common_suffixes'] = common_suffixes
    return d


# --------- MAIN PROGRAM -----------
# --
# -- Usage:  baseline-NER.py target-dir
# --
# -- Extracts Drug NE from all XML files in target-dir, and writes
# -- them in the output format requested by the evalution programs.
# --

# directory with files to process
datadir = sys.argv[1]

# other work
info_lists = get_info_lists()

# process each file in directory
for f in listdir(datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir + "/" + f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value  # get sentence id
        spans = []
        stext = s.attributes["text"].value  # get sentence text
        entities = s.getElementsByTagName("entity")
        for e in entities:
            # for discontinuous entities, we only get the first span
            # (will not work, but there are few of them)
            (start, end) = e.attributes["charOffset"].value.split(";")[0].split("-")
            typ = e.attributes["type"].value
            spans.append((int(start), int(end), typ))

        # convert the sentence to a list of tokens
        tokens = tokenize(stext)
        pos_tags = pos_tag([x for (x, _, _) in tokens])
        # extract sentence features
        features = extract_features(tokens, pos_tags, info_lists)

        # print features in format expected by crfsuite trainer
        for i in range(0, len(tokens)):
            # see if the token is part of an entity
            tag = get_tag(tokens[i], spans)
            print(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

        # blank line to separate sentences
        print()
