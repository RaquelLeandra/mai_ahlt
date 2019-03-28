#!/bin/bash

FINAL_TEST_FILE=task9.1_CRF_cv.txt

rm $FINAL_TEST_FILE

python3 extract-features.py ../../data/Train/All/ > train_all.features

for i in {0..9}
do
    python3 update_feature_files.py $i
    python3 train-crf.py mymodel.crf < train_cv.features
    python3 predict-crf.py mymodel.crf < test_cv.features >>$FINAL_TEST_FILE
done

java -jar ../../eval/evaluateNER.jar ../../data/Train/All $FINAL_TEST_FILE
