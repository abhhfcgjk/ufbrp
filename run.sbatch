#!/bin/bash
#SBATCH --job-name=28i_mel_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=0
#SBATCH --nodelist=cn[1-12,14-19,21-30]


set -e

IMAGE=/scratch/amoskalenko/users/28i_mel/base270.sqsh
WORKDIR=/scratch/amoskalenko/users/28i_mel/ufbrp
LOG=output.log

srun --ntasks=1 -G 8 \
     --container-image "${IMAGE}" \
     --container-mounts "${WORKDIR}:/workspace" \
     bash -lc '
set -e
cd /workspace
apt update
apt install git
pip install -r requirements.txt 
python -m robustbench.eval --dataset=imagenet --model_name=LipReg_aa-adv --eps=0.0156862745 --n_ex=5000 --batch_size=64
' > $LOG 2>&1 
