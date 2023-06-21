#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=WRN16d8d4d4
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=patel.purvi@northeastern.edu


module load cuda/11.7
source /home/patel.purvi/miniconda3/bin/activate
conda activate gdl

python -c'import torch; print(torch.cuda.is_available())'
python /home/patel.purvi/Galaxy-Morphology-Classification/src/train.py --config /home/patel.purvi/Galaxy-Morphology-Classification/config/ViT16.yaml