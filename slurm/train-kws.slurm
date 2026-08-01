#!/bin/bash
#SBATCH --job-name=kws-train
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/%u/voice-kws/logs/train-%x-%j.out
#SBATCH --error=/home-mscluster/%u/voice-kws/logs/train-%x-%j.err

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
module load pytorch/2026-02-24

# --- args below map to your KWS sweep: architecture + b value ---
python src/train_kws.py \
    --arch "$ARCH" \
    --b "$B_VALUE" \
    --data ~/voice-kws/data/speech_commands \
    --checkpoint-dir ~/voice-kws/checkpoints \
    --results-dir ~/voice-kws/results

# --- auto commit + push results so your laptop stays in sync ---
cd $SLURM_SUBMIT_DIR
bash scripts/precommit-check.sh
git add checkpoints/ logs/ results/
git commit -m "Auto-commit: ${SLURM_JOB_NAME} job ${SLURM_JOB_ID} (arch=${ARCH}, b=${B_VALUE})"
git push
