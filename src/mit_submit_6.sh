#!/bin/bash
#SBATCH --job-name=deepfaker              # Name
#SBATCH --gres=gpu:volta:2                # 2 Volta GPUs
#SBATCH --cpus-per-task=40                # 40 CPUs per task
#SBATCH --time=24:00:00                   # Maximum runtime
#SBATCH --mem=340G                        # Memory

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTPUT_DIR="$(pwd)/workflow"
mkdir -p "$OUTPUT_DIR"

OUT_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.out"
ERR_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.err"

exec > "$OUT_FILE"
exec 2> "$ERR_FILE"

# Load necessary modules
source /etc/profile
module load anaconda/Python-ML-2024b

# Notice the space after --strategy "ddp"
srun python3 Demo_6.py \
  --max_epochs 1000000 \
  --ckpt_dir "$(pwd)/../src/ckpt"
  --ckpt_interval 100 \
  --strategy "ddp" \
  --num_nodes 1
