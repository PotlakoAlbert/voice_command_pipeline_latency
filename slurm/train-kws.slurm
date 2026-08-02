#!/bin/bash
#SBATCH --job-name=kws-train
#SBATCH --partition=biggpu
#SBATCH --gres=gpu:1
#SBATCH --output=/home-mscluster/%u/voice_command_pipeline_latency/logs/train-%x-%j.out
#SBATCH --error=/home-mscluster/%u/voice_command_pipeline_latency/logs/train-%x-%j.err

sleep 20

echo "=== Useful Debugging Information ==="
echo "Date: "$(date)
echo "Username: "$USER
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Home Directory: "$HOME
echo "Node: "$HOSTNAME
echo "Current Working Directory: "$PWD
echo "Number of Nodes: "$SLURM_JOB_NUM_NODES
echo "List of Nodes: "$SLURM_JOB_NODELIST
echo "======"

cd $SLURM_SUBMIT_DIR
mkdir -p logs checkpoints results

source /etc/profile.d/lmod.sh
module load anaconda/anaconda-2025-12-2

conda run -n kws python -c "import torch; print('torch:', torch.__version__, '| cuda available:', torch.cuda.is_available())"

conda run -n kws python src/data.py --data ~/voice_command_pipeline_latency/data
