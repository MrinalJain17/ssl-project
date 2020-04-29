#!/bin/bash

# Modified from https://wikis.nyu.edu/display/NYUHPC/Running+Jupyter+on+Prince

#SBATCH --job-name=python_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --gres=gpu:p100:1
#SBATCH --time=12:00:00

module purge
module load cudnn/10.1v7.6.5.32
module load cuda/10.1.105

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

. ~/.bashrc
cd $(pwd)
conda activate base

python road_map_construction.py --EPOCHS 25
