#!/bin/bash

# Modified from https://wikis.nyu.edu/display/NYUHPC/Running+Jupyter+on+Prince

#SBATCH --job-name=python_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

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

# python road_map_construction.py --NUM_LAYERS 2 --FEATURES_START 256 --DROPOUT 0.25 --LOSS bce --VERSION 0
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 64 --DROPOUT 0.1 --LOSS bce --VERSION 1
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --LOSS bce --VERSION 2
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 256 --DROPOUT 0.4 --LOSS bce --VERSION 3
# 
# python road_map_construction.py --NUM_LAYERS 2 --FEATURES_START 256 --DROPOUT 0.25 --LOSS mse --VERSION 4
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 64 --DROPOUT 0.1 --LOSS mse --VERSION 5
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --LOSS mse --VERSION 6
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 256 --DROPOUT 0.4 --LOSS mse --VERSION 7
# 
# python road_map_construction.py --NUM_LAYERS 2 --FEATURES_START 256 --DROPOUT 0.25 --LOSS mae --VERSION 8
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 64 --DROPOUT 0.1 --LOSS mae --VERSION 9
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --LOSS mae --VERSION 10
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 256 --DROPOUT 0.4 --LOSS mae --VERSION 11
# 
# python road_map_construction.py --NUM_LAYERS 2 --FEATURES_START 256 --DROPOUT 0.25 --LOSS psnr_mse --VERSION 12
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 64 --DROPOUT 0.1 --LOSS psnr_mse --VERSION 13
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --LOSS psnr_mse --VERSION 14
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 256 --DROPOUT 0.4 --LOSS psnr_mse --VERSION 15
# 
# python road_map_construction.py --NUM_LAYERS 2 --FEATURES_START 256 --DROPOUT 0.25 --LOSS psnr_mae --VERSION 16
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 64 --DROPOUT 0.1 --LOSS psnr_mae --VERSION 17
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --LOSS psnr_mae --VERSION 18
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 256 --DROPOUT 0.4 --LOSS psnr_mae --VERSION 19

# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --LOSS bce+mse --VERSION 20
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --LOSS weighted_bce --VERSION 21

# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --LOSS weighted_bce --VERSION 22

# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --LOSS dice_loss --VERSION 23
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.25 --LOSS dice_loss --LEARNING_RATE 0.1 --VERSION 24
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.40 --LOSS dice_loss --LEARNING_RATE 0.1 --VERSION 25
# python road_map_construction.py --NUM_LAYERS 2 --FEATURES_START 256 --DROPOUT 0.40 --LOSS dice_loss --LEARNING_RATE 0.1 --VERSION 26
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 256 --DROPOUT 0.50 --LOSS dice_loss --LEARNING_RATE 0.1 --VERSION 27
# python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 256 --DROPOUT 0.50 --LOSS dice_loss --LEARNING_RATE 0.1 --EPOCHS 100 --VERSION 28

python road_map_construction.py --NUM_LAYERS 3 --FEATURES_START 128 --DROPOUT 0.40 --LOSS dice_loss --LEARNING_RATE 0.1 --EPOCHS 100 --VERSION 29
