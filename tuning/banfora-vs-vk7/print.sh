#!/bin/bash

directory="banfora-vs-vk7/xgboost/"

files=$(ls -v "$directory")

for file in $files; do
    wc -l "$directory/$file"
done 


