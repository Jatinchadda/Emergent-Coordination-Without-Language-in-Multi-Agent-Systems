from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.envs.intersection_env import IntersectionEnv
from src.agents.ppo import train_ppo


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

TOTAL_STEPS = 100_000

# Change ONLY these two values for a new training run.
SEED = 50
MODEL_NAME = "conservative_seed50.pt"

MODEL_PATH = ROOT / "models" / MODEL_NAME


# ============================================================
# ENVIRONMENT
# ============================================================

env = IntersectionEnv(
    behavior="conservative",
    seed=SEED,
    max_steps=600,
)


# ============================================================
# PPO TRAINING
# ============================================================

model, training_stats = train_ppo(
    env=env,
    total_steps=TOTAL_STEPS,
    seed=SEED,
    save_path=str(MODEL_PATH),
    rollout_size=2048,
    batch_size=256,
    update_epochs=6,
)

print()
print(f"Training complete.")
print(f"Behavior: conservative")
print(f"Seed: {SEED}")
print(f"Model saved to: {MODEL_PATH}")