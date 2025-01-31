#!/bin/bash
#SBATCH -J deepfaker
#SBATCH -N 16
#SBATCH --ntasks-per-node=4
#SBATCH -p rtx
#SBATCH -t 48:00:00
#SBATCH --exclusive

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/work2/10214/yu_yao/frontera/deepfaker/src/workflow"
mkdir -p "$OUTPUT_DIR"

OUT_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.out"
ERR_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.err"

exec > "$OUT_FILE"
exec 2> "$ERR_FILE"

module load python3/3.9.2
source ~/.bashrc

# Notice the space after --strategy "ddp"
srun python3 Demo_6.py \
  --max_epochs 1000000 \
  --ckpt_dir "/work2/10214/yu_yao/frontera/deepfaker/src/ckpt" \
  --ckpt_interval 1000 \
  --strategy "ddp" \
  --num_nodes 16

