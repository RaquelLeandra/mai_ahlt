#!/bin/bash

python3 extract-features.py ../../data/Train/All > train.features
python3 extract-features.py ../../data/Test-NER/All > test.features
python3 train-crf.py mymodel.crf < train.features > /dev/null
python3 predict-crf.py mymodel.crf <test.features >task9.1_CRF_testset.txt
java -jar ../../eval/evaluateNER.jar ../../data/Test-NER/All task9.1_CRF_testset.txt
