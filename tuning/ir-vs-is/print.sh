#!/bin/bash

directory="ir-vs-is/xgboost/"
files=$(ls -v "$directory")


for file in $files; do
    wc -l "$directory/$file"
done 


