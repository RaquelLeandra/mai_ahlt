from os import listdir
import os
import shutil

TRAIN_DATA_PATH = '../../data/Train/All'
NEW_DATA_PATH = './cv_data'
os.mkdir(NEW_DATA_PATH)
l = listdir(TRAIN_DATA_PATH)
num_folds = 10
fold_size = len(l) // num_folds
folds = [l[i * fold_size:(i + 1) * fold_size] for i in range(10)]

for i in range(num_folds):
    fold_path = 'fold' + str(i)
    os.mkdir(os.path.join(NEW_DATA_PATH, fold_path))
    # test folder
    os.mkdir(os.path.join(NEW_DATA_PATH, fold_path, 'test'))
    test_files = folds[i]
    for file_name in test_files:
        src = os.path.join(TRAIN_DATA_PATH, file_name)
        dst = os.path.join(NEW_DATA_PATH, fold_path, 'test')
        shutil.copy(src, dst)
    # train folder
    os.mkdir(os.path.join(NEW_DATA_PATH, fold_path, 'train'))
    train_folds = folds[:i] + folds[i + 1:]
    train_files = [x for j in range(num_folds) if j != i for x in folds[j]]
    for file_name in train_files:
        src = os.path.join(TRAIN_DATA_PATH, file_name)
        dst = os.path.join(NEW_DATA_PATH, fold_path, 'train')
        shutil.copy(src, dst)
