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


EPISODES = 100
START_SEED = 3000

MODELS = {
    "aggressive": {
        "behavior": "aggressive",
        "path": ROOT / "models" / "aggressive_seed45.pt",
    },

    "neutral": {
        "behavior": "neutral",
        "path": ROOT / "models" / "neutral_seed46.pt",
    },

    "conservative": {
        "behavior": "conservative",
        "path": ROOT / "models" / "conservative_seed50.pt",
    },
}

ACTION_NAMES = [
    "Brake",
    "Yield",
    "Maintain",
    "Accelerate",
]


def choose_action(
    model,
    obs,
):

    normalized_obs = (
        normalize_for_model(
            model,
            obs,
        )
    )

    obs_tensor = torch.tensor(
        normalized_obs,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        logits, _ = model.forward(
            obs_tensor
        )

        action = int(
            torch.argmax(
                logits,
                dim=-1,
            ).item()
        )

    return action


def evaluate_behavior(
    behavior,
    model,
    seeds,
):

    results = []

    action_counts = np.zeros(
        4,
        dtype=np.int64,
    )

    for episode_index, seed in enumerate(
        seeds,
        start=1,
    ):

        env = IntersectionEnv(
            behavior=behavior,
            seed=seed,
            max_steps=600,
        )

        obs, _ = env.reset(
            seed=seed
        )

        reward_total = 0.0

        # Accumulate speed across the entire episode.
        speed_sum = 0.0
        speed_samples = 0

        collision = False
        completed = False

        while True:

            action = choose_action(
                model,
                obs,
            )

            if not 0 <= action <= 3:
                raise RuntimeError(
                    f"Invalid action {action}"
                )

            action_counts[action] += 1

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            reward_total += float(
                reward
            )

            # Record speed at every simulation step.
            speed_sum += float(
                env.learning_vehicle.speed
            )

            speed_samples += 1

            collision |= bool(
                info.get(
                    "collision",
                    False,
                )
            )

            completed |= bool(
                info.get(
                    "completed",
                    False,
                )
            )

            if terminated or truncated:
                break

        completed |= bool(
            env.learning_vehicle.completed
        )

        # Mean speed over the entire episode.
        average_speed = (
            speed_sum / speed_samples
            if speed_samples > 0
            else 0.0
        )

        results.append(
            {
                "reward": reward_total,
                "collision": collision,
                "completed": completed,
                "speed": average_speed,
                "steps": env.step_count,
                "near_collisions": env.near_collisions,
                "minimum_distance": env.minimum_distance_seen,
                "minimum_ttc": env.minimum_ttc_seen,
            }
        )

        if episode_index % 10 == 0:

            print(
                f"  Evaluated "
                f"{episode_index}/{len(seeds)}"
            )

    return (
        results,
        action_counts,
    )


def finite_mean(values):

    values = [
        value
        for value in values
        if np.isfinite(value)
    ]

    if not values:
        return float("inf")

    return float(
        np.mean(values)
    )


print()
print("=" * 70)
print("COMMON-SEED BEHAVIOR COMPARISON")
print("=" * 70)

print(
    f"Episodes: {EPISODES} | "
    f"Seeds: {START_SEED}-"
    f"{START_SEED + EPISODES - 1}"
)

seeds = list(
    range(
        START_SEED,
        START_SEED + EPISODES,
    )
)

summary = {}


for model_name, model_info in MODELS.items():

    behavior = model_info["behavior"]
    model_path = model_info["path"]

    print()
    print(
        f"Evaluating {model_name} "
        f"(behavior={behavior})..."
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model:\n"
            f"{model_path}"
        )

    model = load_model(
        str(model_path)
    )

    model_action_dim = (
        model.actor.out_features
    )

    if model_action_dim != 4:

        raise RuntimeError(
            f"{behavior} model has "
            f"{model_action_dim} actions. "
            f"Expected 4."
        )

    (
        results,
        action_counts,
    ) = evaluate_behavior(
        behavior,
        model,
        seeds,
    )

    total_actions = max(
        int(action_counts.sum()),
        1,
    )

    rates = (
        action_counts
        / total_actions
    )

    summary[model_name] = {

        "collision":
            np.mean(
                [
                    float(r["collision"])
                    for r in results
                ]
            ),

        "completion":
            np.mean(
                [
                    float(r["completed"])
                    for r in results
                ]
            ),

        "reward":
            np.mean(
                [
                    r["reward"]
                    for r in results
                ]
            ),

        # Average speed over all episodes.
        "speed":
            np.mean(
                [
                    r["speed"]
                    for r in results
                ]
            ),

        # Average speed only among successfully
        # completed episodes.
        "completed_speed":
            np.mean(
                [
                    r["speed"]
                    for r in results
                    if r["completed"]
                ]
            )
            if any(
                r["completed"]
                for r in results
            )
            else 0.0,

        "steps":
            np.mean(
                [
                    r["steps"]
                    for r in results
                ]
            ),

        "near_collisions":
            np.mean(
                [
                    r["near_collisions"]
                    for r in results
                ]
            ),

        "minimum_distance":
            finite_mean(
                [
                    r["minimum_distance"]
                    for r in results
                ]
            ),

        "minimum_ttc":
            finite_mean(
                [
                    r["minimum_ttc"]
                    for r in results
                ]
            ),

        "brake":
            rates[0],

        "yield":
            rates[1],

        "maintain":
            rates[2],

        "accelerate":
            rates[3],
    }


print()
print("=" * 100)
print("FINAL COMPARISON")
print("=" * 100)

model_names = list(MODELS.keys())

print(
    f"{'Metric':<32}"
    f"{model_names[0]:>22}"
    f"{model_names[1]:>22}"
    f"{model_names[2]:>22}"
)

print("-" * 100)


def print_row(label, key):

    print(
        f"{label:<32}"
        f"{summary[model_names[0]][key]:>22.3f}"
        f"{summary[model_names[1]][key]:>22.3f}"
        f"{summary[model_names[2]][key]:>22.3f}"
    )


print_row(
    "Collision",
    "collision",
)

print_row(
    "Completion",
    "completion",
)

print_row(
    "Reward",
    "reward",
)

print_row(
    "Average speed (m/s)",
    "speed",
)

print_row(
    "Completed-episode speed (m/s)",
    "completed_speed",
)

print_row(
    "Steps",
    "steps",
)

print_row(
    "Near collisions",
    "near_collisions",
)

print_row(
    "Min distance (m)",
    "minimum_distance",
)

print_row(
    "Min TTC (s)",
    "minimum_ttc",
)

print()
print("ACTION DISTRIBUTION")
print("-" * 100)

print_row(
    "Brake rate",
    "brake",
)

print_row(
    "Yield rate",
    "yield",
)

print_row(
    "Maintain rate",
    "maintain",
)

print_row(
    "Accelerate rate",
    "accelerate",
)

print()
print("=" * 100)
print("COMPARISON COMPLETE")
print("=" * 100)