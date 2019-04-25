#!/usr/bin/env bash

output_path_name="task9.2_raquel_0.txt"

m "evaluations/$output_path_name"

python3 baseline-DDI.py \
    ../../data/Test-DDI/DrugBank > evaluations/$output_path_name

java -jar ../../eval/evaluateDDI.jar ../../data/Test-DDI/DrugBank "evaluations/$output_path_name"