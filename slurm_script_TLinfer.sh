#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --mem 16G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH -C 'gpu32g'

source activate pygpu1


nnUNet_predict -i /cluster/projects/radiomics/Gyn_Autosegmentation/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task046_gyn/imagesTs -o /cluster/projects/radiomics/Gyn_Autosegmentation/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task046_gyn/infersTs_scratch/infersTs_scratch25 -t 46 -m 3d_fullres -tr nnUNetTrainerV2_epoch50 -f all -p nnUNetPlans_pretrained_TL
