from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.envs.intersection_env import IntersectionEnv
from src.agents.ppo import (
    load_model,
    normalize_for_model,
)


MODEL_PATH = ROOT / "models" / "aggressive.pt"


env = IntersectionEnv(
    behavior="aggressive",
    seed=2024,
    max_steps=600,
)

model = load_model(MODEL_PATH)

obs, _ = env.reset(seed=2024)

print("=" * 70)
print("OBSERVATION / POLICY DEBUG")
print("=" * 70)

for step in [0, 1, 5, 10, 20, 30, 40]:

    while env.step_count < step:
        obs, reward, terminated, truncated, info = env.step(1)

        if terminated or truncated:
            break

    normalized_obs = normalize_for_model(
    model,
    obs
    )

    obs_tensor = torch.tensor(
        normalized_obs,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(obs_tensor)
        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0].numpy()

    print()
    print(f"STEP {env.step_count}")
    print("-" * 50)

    print("Action probabilities:")
    print(
        f"Brake      : {probabilities[0]:.4f}"
    )
    print(
        f"Maintain   : {probabilities[1]:.4f}"
    )
    print(
        f"Accelerate : {probabilities[2]:.4f}"
    )

    print()
    print("Observation:")
    print(np.round(obs, 4))

    print()
    print("Environment state:")
    print(
        f"Speed            : "
        f"{env.learning_vehicle.speed:.3f}"
    )
    print(
        f"Acceleration     : "
        f"{env.learning_vehicle.acceleration:.3f}"
    )
    print(
        f"Minimum distance : "
        f"{env.minimum_distance_seen:.3f}"
    )
    print(
        f"Minimum TTC      : "
        f"{env.minimum_ttc_seen:.3f}"
    )

    print(
        f"Vehicles         : "
        f"{len(env.vehicles)}"
    )

print()
print("=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)