from os import listdir
import os
import shutil
import sys

ALL_FEATURES_FILE = 'train_all.features'
cv_folds = 5

with open(ALL_FEATURES_FILE) as f:
    l = f.readlines()

# split list in cv_folds parts
chunk_size = len(l) // cv_folds
l = [l[i * chunk_size:(i + 1) * chunk_size] for i in range(cv_folds)]

test_fold_index = int(sys.argv[1])

with open('train_cv.features', 'w') as f:
    for i in range(cv_folds):
        if i != test_fold_index:
            for row in l[i]:
                f.write(row)

with open('test_cv.features', 'w') as f:
    for i in range(cv_folds):
        if i == test_fold_index:
            for row in l[i]:
                f.write(row)
