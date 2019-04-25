# LSTM-CRF for sequence tagging using keras
# reference: https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/



# read csv data set and transform it in order to feed to the network
# example:
# input -> [The, benefits, of, waxanoryl, should, be, ...]
# output -> [O, O, O, B-drug, O, O, ...]



# to facilitate input to the network, create word index and pad sequences





# train the model






# predict the name entities in the test set
# y_pred = ...


# transform the predictions into a file in the evaluation format
# example: DDI-MedLine.d203.s1|129-143|antidepressants|group
# iterate through sentences of test data set and for each one look at the predictions.
# when one named entity is found, look at the char offset in the data frame, and
# construct the line sentence_id|char_offset|text|entity_type







# run evaulation script with "subprocess"
