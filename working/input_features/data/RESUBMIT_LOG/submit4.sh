#!/bin/bash
# This is an example of basis set projection
#SBATCH --job-name=chmbl1
#SBATCH --output=slurm.out
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --account=davinci
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

source activate cytoxnet

python script4.py >> outall.dat
