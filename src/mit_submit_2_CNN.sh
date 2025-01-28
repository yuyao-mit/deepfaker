#!/bin/bash
#SBATCH --job-name=demo_DF_2              # Name
#SBATCH --output=/home/gridsan/yyao/Research_Projects/Microstructure_Enough/deep_faker/demo/demo_log/demo_DF_2%j.log   # Output
#SBATCH --gres=gpu:volta:2                # 2 Volta GPUs
#SBATCH --cpus-per-task=40                # 40 CPUs per task
#SBATCH --time=24:00:00                   # Maximum runtime
#SBATCH --mem=340G                        # Memory

# Load necessary modules
source /etc/profile
module load anaconda/Python-ML-2024b

# Run the Python script
python Demo_DF_2.py --input_dim 128

