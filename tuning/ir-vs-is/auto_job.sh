#!/bin/bash

for job_number in {0..9}; do
    sbatch_file="jobs/sbatch_job${job_number}.sh"

    cat > "$sbatch_file" <<EOF
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

python3 -u ir-vs-is/tuning_multiclass.py $job_number > jobs-output/job_output$job_number.out

exit 0
EOF

    sbatch "$sbatch_file"

    echo "Submitted job $job_number"
done

