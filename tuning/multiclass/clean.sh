#!/bin/bash

directory="/home/eng/esrdfn/tuning/multiclasss/"

find "$directory" -type f -name 'slurm-[0-9][0-9][0-9][0-9][0-9][0-9].out' -exec rm {} \;

