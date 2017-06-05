#!/bin/bash
#
#BATCH -J single_fouriernetwork
#SBATCH -o single_fouriernetwork32.out
#SBATCH -e single_fouriernetwork32.err
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 0-40:00
#SBATCH -p serial_requeue
#SBATCH --mem=24000
#SBATCH --mail-type=ALL      
#SBATCH --mail-user=mnye@college.harvard.edu

module load gcc/4.9.3-fasrc01 tensorflow/0.12.0-fasrc02
python fouriernetwork_odyssey.py
