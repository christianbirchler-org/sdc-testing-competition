#!/bin/bash

for i in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200; do
    for j in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200; do
        python evaluation.py -u localhost:5454 --sample-size $i --subject-size $j
    done
done
