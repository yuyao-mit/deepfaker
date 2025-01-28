#!/bin/bash
#SBATCH -J deep_faker_train
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH -p rtx
#SBATCH -t 48:00:00
#SBATCH --exclusive

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/work2/10214/yu_yao/frontera/deep_faker/work_flow"
mkdir -p "$OUTPUT_DIR"
OUT_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.out"
ERR_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.err"
exec > "$OUT_FILE"
exec 2> "$ERR_FILE"

module load python3/3.9.2
source ~/.bashrc

srun python3 Demo_4_RES_CNN_1.py --num_nodes 8
