# Emergent Coordination Without Language

Research project on coordination between heterogeneous driving agents at an unsignalized intersection, without explicit communication.

The repository contains independently trained Stage 1 behavior policies and a Stage 2 three-agent MAPPO environment. Stage 2 uses decentralized actors with a centralized training critic; background traffic is enabled throughout MAPPO training and evaluation.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

Run the Stage 2 environment regressions:

```bash
python src/stage2/test_stage2.py
```

Train Stage 2 MAPPO:

```bash
python src/stage2/training/train_mappo.py --total-updates 50 --eval-interval 10 --eval-episodes 5 --final-eval-episodes 20
```

Add `--device cuda` or `--device cpu` to choose a device. Training writes checkpoints and evaluation reports under `models/stage2_mappo/` by default.

## Project Layout

- `src/agents/` contains Stage 1 behavior profiles and PPO implementation.
- `src/envs/` contains the Stage 1 driving environment.
- `src/stage2/` contains the intersection environment, diagnostics, tests, and MAPPO trainer.
- `models/` contains selected Stage 1 policies and Stage 2 checkpoints.