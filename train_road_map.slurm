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

python road_map_construction.py --NUM_LAYERS 2 --FEATURES_START 64 --DROPOUT 0.1 --VERSION 0
python road_map_construction.py --NUM_LAYERS 2 --FEATURES_START 64 --DROPOUT 0.1 --USE_MSE --VERSION 1

python road_map_construction.py --NUM_LAYERS 2 --FEATURES_START 256 --DROPOUT 0.25 --VERSION 2
python road_map_construction.py --NUM_LAYERS 2 --FEATURES_START 256 --DROPOUT 0.25 --USE_MSE --VERSION 3

python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 64 --DROPOUT 0.1 --VERSION 4
python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 64 --DROPOUT 0.1 --USE_MSE --VERSION 5

python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --VERSION 6
python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --USE_MSE --VERSION 7

python road_map_construction.py --NUM_LAYERS 4 --FEATURES_START 64 --DROPOUT 0.2 --VERSION 8
python road_map_construction.py --NUM_LAYERS 4 --FEATURES_START 64 --DROPOUT 0.2 --USE_MSE --VERSION 9
