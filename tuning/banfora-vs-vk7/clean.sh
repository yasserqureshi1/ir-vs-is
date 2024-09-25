#!/bin/bash

directory="banfora-vs-vk7/"
find "$directory" -type f -name 'slurm-[0-9][0-9][0-9][0-9][0-9][0-9].out' -exec rm {} \;

