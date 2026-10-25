#!/bin/bash
#SBATCH -p batch
#SBATCH -J mwf-compare
#SBATCH -o logs/mwf-compare-%j.out
#SBATCH -t 00:30:00
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/mini_wf_compare.py
