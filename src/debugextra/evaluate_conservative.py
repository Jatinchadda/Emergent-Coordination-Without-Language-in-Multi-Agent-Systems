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


MODEL_PATH = ROOT / "models" / "conservative.pt"

NUM_EPISODES = 100
BASE_SEED = 2000


def evaluate_episode(model, seed):

    env = IntersectionEnv(
        behavior="conservative",
        seed=seed,
        max_steps=600,
    )

    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    actions = []
    speeds = []

    collision = False

    for _ in range(env.max_steps):

        normalized_obs = normalize_for_model(
            model,
            obs,
        )

        obs_tensor = torch.tensor(
            normalized_obs,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(obs_tensor)

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0].cpu().numpy()

        action = int(
            np.argmax(probabilities)
        )

        actions.append(action)

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        speeds.append(
            info["speed"]
        )

        if info["collision"]:
            collision = True

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "collision": int(collision),
        "completed": int(
            env.learning_vehicle.completed
        ),
        "speed": float(
            np.mean(speeds)
        ),
        "steps": env.step_count,
        "near_collisions": env.near_collisions,
        "minimum_distance": env.minimum_distance_seen,
        "minimum_ttc": env.minimum_ttc_seen,
        "accelerate_rate": float(
            np.mean(
                np.asarray(actions) == 2
            )
        ),
        "maintain_rate": float(
            np.mean(
                np.asarray(actions) == 1
            )
        ),
        "brake_rate": float(
            np.mean(
                np.asarray(actions) == 0
            )
        ),
    }


print("=" * 60)
print("CONSERVATIVE MODEL EVALUATION")
print("=" * 60)

print(f"Model: {MODEL_PATH}")
print(f"Episodes: {NUM_EPISODES}")
print()

model = load_model(MODEL_PATH)

results = []

for episode in range(NUM_EPISODES):

    result = evaluate_episode(
        model,
        BASE_SEED + episode,
    )

    results.append(result)

    if (episode + 1) % 10 == 0:
        print(
            f"Evaluated "
            f"{episode + 1}/{NUM_EPISODES} episodes"
        )


collision_rate = np.mean([
    r["collision"]
    for r in results
])

completion_rate = np.mean([
    r["completed"]
    for r in results
])

average_reward = np.mean([
    r["reward"]
    for r in results
])

average_speed = np.mean([
    r["speed"]
    for r in results
])

average_steps = np.mean([
    r["steps"]
    for r in results
])

average_near_collisions = np.mean([
    r["near_collisions"]
    for r in results
])

finite_distances = [
    r["minimum_distance"]
    for r in results
    if np.isfinite(r["minimum_distance"])
]

average_min_distance = (
    float(np.mean(finite_distances))
    if finite_distances
    else float("nan")
)

finite_ttc = [
    r["minimum_ttc"]
    for r in results
    if np.isfinite(r["minimum_ttc"])
]

average_min_ttc = (
    float(np.mean(finite_ttc))
    if finite_ttc
    else float("nan")
)

accelerate_rate = np.mean([
    r["accelerate_rate"]
    for r in results
])

maintain_rate = np.mean([
    r["maintain_rate"]
    for r in results
])

brake_rate = np.mean([
    r["brake_rate"]
    for r in results
])


print()
print("=" * 60)
print("FINAL CONSERVATIVE RESULTS")
print("=" * 60)

print(
    f"Collision rate       : "
    f"{collision_rate:.3f}"
)

print(
    f"Completion rate      : "
    f"{completion_rate:.3f}"
)

print(
    f"Average reward       : "
    f"{average_reward:.3f}"
)

print(
    f"Average speed        : "
    f"{average_speed:.3f} m/s"
)

print(
    f"Average steps        : "
    f"{average_steps:.1f}"
)

print(
    f"Near collisions/ep   : "
    f"{average_near_collisions:.3f}"
)

print(
    f"Average minimum dist : "
    f"{average_min_distance:.3f} m"
)

print(
    f"Average minimum TTC  : "
    f"{average_min_ttc:.3f} s"
)

print()
print("ACTION DISTRIBUTION")
print("-" * 30)

print(
    f"Accelerate rate      : "
    f"{accelerate_rate:.3f}"
)

print(
    f"Maintain rate        : "
    f"{maintain_rate:.3f}"
)

print(
    f"Brake rate           : "
    f"{brake_rate:.3f}"
)

print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)