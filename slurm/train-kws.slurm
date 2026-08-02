#!/bin/bash
#SBATCH --job-name=kws-train
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/%u/voice_command_pipeline_latency/logs/train-%x-%j.out
#SBATCH --error=/home-mscluster/%u/voice_command_pipeline_latency/logs/train-%x-%j.err

sleep 20

echo "=== Useful Debugging Information ==="
echo "Date: "$(date)
echo "Username: "$USER
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Node: "$HOSTNAME
echo "======"

cd $SLURM_SUBMIT_DIR
mkdir -p logs checkpoints results

echo "=== GPU Diagnostics ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "======"

source /etc/profile.d/lmod.sh
module load anaconda/anaconda-2025-12-2
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9

conda run -n kws python -c "import torch; print('torch:', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())"

conda run -n kws python src/data.py --data ~/voice_command_pipeline_latency/data

echo "=== Extra CUDA Diagnostics ==="
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
conda run -n kws python -c "
import torch
print('torch cuda build version:', torch.version.cuda)
try:
    torch.cuda.init()
    print('cuda.init() succeeded')
except Exception as e:
    print('cuda.init() FAILED:', repr(e))
print('device count:', torch.cuda.device_count())
"
ls -la /dev/nvidia* 2>&1
echo "======"

echo "=== libcuda search ==="
find / -name "libcuda.so*" 2>/dev/null
ldconfig -p | grep -i cuda
echo "======"
