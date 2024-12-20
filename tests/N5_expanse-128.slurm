#!/bin/bash
#SBATCH --job-name="../wave260"
#SBATCH --output="N5_128.%j.%N.out"
#SBATCH --partition=compute
# #SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
# #SBATCH --mem=512M
#SBATCH --mem=40G
#SBATCH --account=csd911
# #SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:03:00
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu

# setup your environment

export SLURM_EXPORT_ENV=ALL
module purge
module load cpu/0.15.4  gcc/10.2.0  mvapich2/2.3.6
# module load cpu/0.15.4  gcc/10.2.0  openmpi/4.0.4
module load netcdf-c/4.7.4
#Load module file(s) into the shell environment
module load slurm
mkdir -p /expanse/lustre/scratch/bwchin/temp_project/job_$SLURM_JOB_ID
srun --chdir /expanse/lustre/scratch/bwchin/temp_project/job_$SLURM_JOB_ID --mpi=pmi2 -n 128 $SLURM_SUBMIT_DIR/../build_expanse/wave260mpi $SLURM_SUBMIT_DIR/../tests/test5500.config  -x 16 -y 8
