#!/bin/bash
#SBATCH --job-name="../wave260"
#SBATCH --output="wave.%j.%N.out"
# #SBATCH --partition=compute
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
# #SBATCH --mem=512M
#SBATCH --mem=1024M
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
srun --chdir /expanse/lustre/scratch/bwchin/temp_project/job_$SLURM_JOB_ID --mpi=pmi2 -n 8 ~/pa3-wave/twoDwave-dev/build_expanse/wave260mpi -n 1000 -i 3000 -x 4 -y 2
srun --chdir /expanse/lustre/scratch/bwchin/temp_project/job_$SLURM_JOB_ID --mpi=pmi2 -n 8 ~/pa3-wave/twoDwave-dev/build_expanse/wave260mpi -n 1000 -i 3000 -x 2 -y 4
# srun --mpi=pmi2 -n 8 ../wave260 -n 2800 -i 2000 -x 2  -y 4
