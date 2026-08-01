# voice-kws

End-to-End Latency Optimisation in Voice Command Pipelines Using Lightweight Models and Keyword Spotting.

## Structure
- `src/`         — pipeline and training code
- `configs/`     — experiment configs (KWS architecture, b sweep values, etc.)
- `slurm/`       — cluster job scripts
- `checkpoints/` — trained model weights (plain git, small models only — see scripts/precommit-check.sh)
- `logs/`        — slurm stdout/stderr and training logs
- `results/`     — metrics, Pareto curve data
- `data/`        — gitignored, reconstructed via corpus scripts

## Workflow
Run jobs on cluster -> commit + push from cluster -> pull on laptop.
