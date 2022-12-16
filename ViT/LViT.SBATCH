#!/bin/bash

#SBATCH --job-name=train_effViT

#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

#SBATCH --time=1-00:00:00
#SBATCH --output=effViT.out
#SBATCH --error=effViT.err




singularity exec --nv \
--overlay /scratch/zz1706/my_env/overlay-15GB-500K.ext3:ro \
/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; python effViT.py"