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


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = (
    ROOT
    / "models"
    / "aggressive_calibrate.pt"
)

NUM_EPISODES = 100
BASE_SEED = 1000


# =========================================================
# ACTION DEFINITIONS
# =========================================================

BRAKE = 0
YIELD = 1
MAINTAIN = 2
ACCELERATE = 3


# =========================================================
# EVALUATE ONE EPISODE
# =========================================================

def evaluate_episode(
    model,
    seed,
):

    env = IntersectionEnv(
        behavior="aggressive",
        seed=seed,
        max_steps=600,
    )

    obs, _ = env.reset(
        seed=seed
    )

    total_reward = 0.0

    actions = []
    speeds = []

    collision = False

    while True:

        # -------------------------------------------------
        # Normalize observation exactly as during training
        # -------------------------------------------------

        normalized_obs = normalize_for_model(
            model,
            obs,
        )

        obs_tensor = torch.tensor(
            normalized_obs,
            dtype=torch.float32,
        ).unsqueeze(0)

        # -------------------------------------------------
        # Deterministic evaluation
        # -------------------------------------------------

        with torch.no_grad():

            logits, _ = model(
                obs_tensor
            )

            action = int(
                torch.argmax(
                    logits,
                    dim=-1,
                ).item()
            )

        # -------------------------------------------------
        # Validate action
        # -------------------------------------------------

        if action not in (
            BRAKE,
            YIELD,
            MAINTAIN,
            ACCELERATE,
        ):

            raise RuntimeError(
                f"Invalid action {action}. "
                f"Expected 0, 1, 2, or 3."
            )

        actions.append(action)

        # -------------------------------------------------
        # Environment step
        # -------------------------------------------------

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += float(
            reward
        )

        speeds.append(
            float(
                info.get(
                    "speed",
                    env.learning_vehicle.speed,
                )
            )
        )

        if info.get(
            "collision",
            False,
        ):
            collision = True

        if terminated or truncated:
            break

    # =====================================================
    # ACTION RATES
    # =====================================================

    action_array = np.asarray(
        actions,
        dtype=np.int64,
    )

    total_actions = max(
        len(action_array),
        1,
    )

    brake_rate = float(
        np.sum(
            action_array == BRAKE
        )
        / total_actions
    )

    yield_rate = float(
        np.sum(
            action_array == YIELD
        )
        / total_actions
    )

    maintain_rate = float(
        np.sum(
            action_array == MAINTAIN
        )
        / total_actions
    )

    accelerate_rate = float(
        np.sum(
            action_array == ACCELERATE
        )
        / total_actions
    )

    # =====================================================
    # RESULT
    # =====================================================

    return {

        "reward": total_reward,

        "collision": int(
            collision
        ),

        "completed": int(
            env.learning_vehicle.completed
        ),

        "speed": float(
            np.mean(speeds)
            if speeds
            else 0.0
        ),

        "steps": int(
            env.step_count
        ),

        "near_collisions": int(
            env.near_collisions
        ),

        "minimum_distance": float(
            env.minimum_distance_seen
        ),

        "minimum_ttc": float(
            env.minimum_ttc_seen
        ),

        "brake_rate": brake_rate,

        "yield_rate": yield_rate,

        "maintain_rate": maintain_rate,

        "accelerate_rate": accelerate_rate,
    }


# =========================================================
# MAIN
# =========================================================

print()
print("=" * 60)
print("AGGRESSIVE CALIBRATION MODEL EVALUATION")
print("=" * 60)

print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Episodes: {NUM_EPISODES}"
)

print(
    "Actions:"
)

print(
    "  0 = Brake"
)

print(
    "  1 = Yield"
)

print(
    "  2 = Maintain"
)

print(
    "  3 = Accelerate"
)

print()


# =========================================================
# MODEL CHECK
# =========================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


model = load_model(
    MODEL_PATH
)


# =========================================================
# VERIFY 4-ACTION CHECKPOINT
# =========================================================

if model.action_dim != 4:

    raise RuntimeError(
        "This checkpoint is not a 4-action model.\n"
        f"Checkpoint action_dim = "
        f"{model.action_dim}\n"
        "Expected action_dim = 4.\n\n"
        "Delete the old checkpoint and retrain "
        "the aggressive calibration model."
    )


# =========================================================
# RUN EVALUATION
# =========================================================

results = []

for episode in range(
    NUM_EPISODES
):

    result = evaluate_episode(
        model,
        BASE_SEED + episode,
    )

    results.append(
        result
    )

    if (
        episode + 1
    ) % 10 == 0:

        print(
            f"Evaluated "
            f"{episode + 1}/"
            f"{NUM_EPISODES} episodes"
        )


# =========================================================
# AGGREGATION
# =========================================================

collision_rate = np.mean(
    [
        r["collision"]
        for r in results
    ]
)

completion_rate = np.mean(
    [
        r["completed"]
        for r in results
    ]
)

average_reward = np.mean(
    [
        r["reward"]
        for r in results
    ]
)

average_speed = np.mean(
    [
        r["speed"]
        for r in results
    ]
)

average_steps = np.mean(
    [
        r["steps"]
        for r in results
    ]
)

average_near_collisions = np.mean(
    [
        r["near_collisions"]
        for r in results
    ]
)


# =========================================================
# MIN DISTANCE
# =========================================================

finite_distances = [
    r["minimum_distance"]
    for r in results
    if np.isfinite(
        r["minimum_distance"]
    )
]

average_min_distance = (
    float(
        np.mean(
            finite_distances
        )
    )
    if finite_distances
    else float("nan")
)


# =========================================================
# MIN TTC
# =========================================================

finite_ttc = [
    r["minimum_ttc"]
    for r in results
    if np.isfinite(
        r["minimum_ttc"]
    )
]

average_min_ttc = (
    float(
        np.mean(
            finite_ttc
        )
    )
    if finite_ttc
    else float("nan")
)


# =========================================================
# ACTION DISTRIBUTION
# =========================================================

brake_rate = np.mean(
    [
        r["brake_rate"]
        for r in results
    ]
)

yield_rate = np.mean(
    [
        r["yield_rate"]
        for r in results
    ]
)

maintain_rate = np.mean(
    [
        r["maintain_rate"]
        for r in results
    ]
)

accelerate_rate = np.mean(
    [
        r["accelerate_rate"]
        for r in results
    ]
)


# =========================================================
# FINAL RESULTS
# =========================================================

print()
print("=" * 60)
print("FINAL AGGRESSIVE CALIBRATION RESULTS")
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
print("-" * 35)

print(
    f"Brake rate           : "
    f"{brake_rate:.3f}"
)

print(
    f"Yield rate           : "
    f"{yield_rate:.3f}"
)

print(
    f"Maintain rate        : "
    f"{maintain_rate:.3f}"
)

print(
    f"Accelerate rate      : "
    f"{accelerate_rate:.3f}"
)

print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)