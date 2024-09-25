#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00

module purge
module load GCCcore/11.2.0 Python/3.9.6

pip3 install --upgrade pip
pip3 install pandas numpy imbalanced-learn scipy scikit-learn xgboost statsmodels joblib
pip3 install --upgrade threadpoolctl

python3 -u banfora-vs-vk7/create_folds.py > jobs-output/cf.out

exit 0