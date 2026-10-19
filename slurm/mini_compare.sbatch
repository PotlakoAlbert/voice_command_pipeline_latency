#!/bin/bash
#SBATCH -p batch
#SBATCH -J mini-compare
#SBATCH -o logs/mini-compare-%j.out
#SBATCH -t 00:20:00
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/mini_compare.py
