#!/bin/bash
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --comment=STEGO_V100

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=34G
#SBATCH --time=7-00:00:00

#SBATCH --output=slurm-output/%j.log
#SBATCH --mail-type=TIME_LIMIT_90,BEGIN,FAIL,END
#SBATCH --mail-user=fwilliams2@sheffield.ac.uk

source .bashrc

# Load the modules
module load Anaconda3/5.3.0 CUDAcore/11.0.2

source setenv_nvcc
source activate stego_nvcc

now=`date`
printf "%s\n" "$now"


python src/train_segmentation.py

now=`date`
printf "%s\n" "$now"


