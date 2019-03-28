#!/usr/bin/env bash

output_path_name="task9.2_raquel_0.txt"

python3 baseline-DDI.py \
    '../../data/Test-DDI/MedLine' > "evaluations/$output_path_name"

java -jar ../../eval/evaluateDDI.jar ../../data/Test-DDI/MedLine "evaluations/$output_path_name"