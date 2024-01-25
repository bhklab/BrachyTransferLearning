#!/bin/bash
#SBATCH -t 30:00:00
#SBATCH --mem 32G
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH -C 'gpu32g'
 

source activate pygpu1

nnUNet_train 3d_fullres nnUNetTrainerV2_epoch50 46 2 -p nnUNetPlans_pretrained_TL 