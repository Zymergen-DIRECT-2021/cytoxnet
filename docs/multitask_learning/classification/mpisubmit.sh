#!/bin/bash
## Job Name
#SBATCH --job-name=hyperopt_graph_c
## Allocation Definit
## The account and partition options should be the same except in a few cases (e.g. ckpt queue and genpool queue).
#SBATCH --account=davinci
#SBATCH --partition=compute
## Resources
## Total number of Nodes
#SBATCH --nodes=1
## Number of cores per node
#SBATCH --ntasks-per-node=8
## Walltime (3 hours). Do not specify a walltime substantially more than your job needs.
#SBATCH --time=48:00:00
## Memory per node. It is important to specify the memory since the default memory is very small.
#SBATCH --mem=60G
## Specify the working directory for this job
##turn on e-mail notification
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=evankomp@uw.edu
## export all your environment variables to the batch job session
#SBATCH --export=all
#SBATCH --output=graph_c_opt.out
source ~/.bash_profile
module load ompi
conda activate cytoxnet
mpirun python -u $1 > $1.dat
