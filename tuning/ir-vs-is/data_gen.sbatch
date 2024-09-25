#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00

module purge
module load GCCcore/11.2.0 Python/3.9.6

pip3 install --upgrade pip
pip3 install pandas numpy scikit-learn joblib
pip3 install --upgrade threadpoolctl

python3 -u /home/eng/esrdfn/tuning/ir-vs-is/data_gen.py > jobs-output/data_gen_output.out

exit 0