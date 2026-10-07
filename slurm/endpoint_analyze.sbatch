#!/bin/bash
#SBATCH -p batch
#SBATCH -J ep-analyze
#SBATCH -o logs/ep-analyze-%j.out
#SBATCH -t 01:00:00
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/analyze_endpoint_sweep.py
