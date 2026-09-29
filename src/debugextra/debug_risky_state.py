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
from src.envs.collision import minimum_distance, minimum_ttc


MODEL_PATH = ROOT / "models" / "aggressive_calibrate.pt"

NUM_EPISODES = 200


model = load_model(MODEL_PATH)

print("=" * 70)
print("AGGRESSIVE POLICY - RISK STATE DEBUG")
print("=" * 70)


found = False


for seed in range(5000, 5000 + NUM_EPISODES):

    env = IntersectionEnv(
        behavior="aggressive",
        seed=seed,
        max_steps=600,
    )

    obs, _ = env.reset(seed=seed)

    for _ in range(600):

        min_dist = minimum_distance(
            env.learning_vehicle,
            env.vehicles,
        )

        ttc = minimum_ttc(
            env.learning_vehicle,
            env.vehicles,
        )

        # Look for a genuinely risky state.
        if (
            np.isfinite(ttc)
            and ttc < 3.0
            and min_dist < 12.0
        ):

            normalized_obs = normalize_for_model(model, obs)

            obs_tensor = torch.tensor(
                normalized_obs,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(obs_tensor)

                probabilities = torch.softmax(
                    logits,
                    dim=-1,
                )[0].cpu().numpy()

            print()
            print("=" * 70)
            print("RISKY STATE FOUND")
            print("=" * 70)

            print(
                f"Seed              : {seed}"
            )
            print(
                f"Step              : {env.step_count}"
            )
            print(
                f"Speed             : "
                f"{env.learning_vehicle.speed:.3f} m/s"
            )
            print(
                f"Minimum distance  : "
                f"{min_dist:.3f} m"
            )
            print(
                f"TTC               : "
                f"{ttc:.3f} s"
            )

            print()
            print("ACTION PROBABILITIES")
            print("-" * 40)

            print(
                f"Brake             : "
                f"{probabilities[0]:.4f}"
            )
            print(
                f"Maintain          : "
                f"{probabilities[1]:.4f}"
            )
            print(
                f"Accelerate        : "
                f"{probabilities[2]:.4f}"
            )

            print()
            print("OBSERVATION")
            print(np.round(obs, 4))

            print()
            print("=" * 70)

            found = True
            break

        obs, reward, terminated, truncated, info = env.step(1)

        if terminated or truncated:
            break

    if found:
        break


if not found:
    print()
    print(
        "No state with TTC < 3.0 s and "
        "distance < 12.0 m was found."
    )