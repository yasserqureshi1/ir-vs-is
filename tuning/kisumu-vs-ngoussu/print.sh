#!/bin/bash


directory="kisumu-vs-ngoussu/xgboost/"

files=$(ls -v "$directory")


for file in $files; do
    wc -l "$directory/$file"
done 


