#!/bin/bash

#SBATCH --job-name=Abeta_classifier.job
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --partition=bme_gpu
##SBATCH --workdir= /public/home/ouzx2022/SynDM/Abeta_classifier
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -t 120:00:00

module add  /public/home/ouzx2022/anaconda3/

python  /public/home/ouzx2022/SynDM/Abeta_classifier/main.py

echo -e "JOB NAME:$SLURM_JOB_NAME,Job ID:$SLURM_JOBID,Allocate Nodes:$SLURM_JOB_NODELIST"