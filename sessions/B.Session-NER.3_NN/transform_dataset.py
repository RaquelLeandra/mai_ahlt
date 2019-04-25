# read xml files and transform data to csv
# columns: sentence_id, word_ix, char_offset_start, char_offset_end, word, tag
# tag values: O, B-(type), I-(type)

from xml.dom.minidom import parse
from os import listdir
import pandas as pd

datadir = '../../data/Train/All'

def tokenize_sentence(sid, stext):
    l = []
    i = 0
    word_count = 0
    while i < len(stext):
        start = i
        word_ix = word_count
        while i < len(stext) and stext[i] not in [' ',',',';','.']:
            i += 1
        end = i-1
        if start < end:
            word_count += 1
            # append word info to list
            l.append([sid, word_ix, start, end, stext[start:end+1], 'O'])
        # carry on
        i += 1

    return l


data = []

# process each file in directory
for f in listdir(datadir):
    print(f)
    # parse XML file, obtaining a DOM tree
    tree = parse(datadir + "/" + f)
    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value  # get sentence id
        stext = s.attributes["text"].value  # get sentence text
        # get list of lists (words and their characteristics)
        ts = tokenize_sentence(sid, stext)
        entities = s.getElementsByTagName("entity")
        for e in entities:
            (start, end) = e.attributes["charOffset"].value.split(";")[0].split("-")
            etype = e.attributes["type"].value
            for w in ts:
                if end == w[3] and start != w[2]:
                    w[5] = 'I-' + etype
                elif end == w[3] and start == w[2]:
                    w[5] = 'B-' + etype
        data += ts

# create dataframe and transform it to csv
df = pd.DataFrame(data)
df.to_csv(index=False)
