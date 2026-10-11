#!/bin/bash
#SBATCH -p batch
#SBATCH -J pipe_moon
#SBATCH -o logs/pipe_moon-%A_%a.out
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
cd ~/voice_command_pipeline_latency
PY=~/conda_envs/kws-cu118/bin/python
C=()
for A in crnn:crnn:crnn dilated_resnet:resnet:resnet repcnn:repcnn:repcnn; do
  IFS=: read P Z T <<< "$A"
  C+=("${Z}_latency_v2.json ${T}_b0.0")
  for b in 0.1 0.2 0.33 0.5 1.0; do C+=("${P}_b${b}_latency.json ${T}_b${b}"); done
  for t in 0.0 0.1 0.2 0.3; do C+=("${P}_ep_tl${t}_latency.json ${T}_ep${t}"); done
done
for p in 0.5 1.0; do
  for c in "${C[@]}"; do
    set -- $c
    $PY src/pipeline_moon.py $1 $p $SLURM_ARRAY_TASK_ID 12 $2
  done
done
