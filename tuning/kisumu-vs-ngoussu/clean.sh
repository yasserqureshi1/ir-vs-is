#!/bin/bash

directory="kisumu-vs-ngoussu/"

find "$directory" -type f -name 'slurm-[0-9][0-9][0-9][0-9][0-9][0-9].out' -exec rm {} \;

