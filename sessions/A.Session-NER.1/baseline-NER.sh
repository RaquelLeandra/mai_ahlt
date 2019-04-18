#!/usr/bin/env bash

input_path_name="../../data/Train/MedLine/"
output_path_name="task9.1_raquel_0.txt"

python baseline-NER.py \
    $input_path_name > "evaluations/$output_path_name"

java -jar ../../eval/evaluateNER.jar $input_path_name "evaluations/$output_path_name"

