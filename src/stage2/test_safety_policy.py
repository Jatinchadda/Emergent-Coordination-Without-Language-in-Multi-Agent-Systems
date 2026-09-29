from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from intersection_env import Stage2IntersectionEnv


MAINTAIN = 2
YIELD = 1
BRAKE = 0


def ttc_seconds(value: float) -> float:
    if not np.isfinite(value):
        return float("inf")
    return float(np.clip(value, 0.0, 1.0) * 10.0)


def other_blocks(observation: np.ndarray):
    observation = np.asarray(observation, dtype=np.float32)

    if observation.shape != (47,):
        raise ValueError(f"Expected observation shape (47,), got {observation.shape}")

    return (
        observation[11:19],
        observation[19:27],
    )


def safest_action(agent_id: str, observation: np.ndarray) -> int:
    blocks = other_blocks(observation)
    ego_index = int(agent_id.rsplit("_", 1)[1])

    conflicts = []

    for slot, block in enumerate(blocks):
        conflict = block[7]
        if conflict < 0.5:
            continue

        ego_ttc = ttc_seconds(block[4])
        other_ttc = ttc_seconds(block[5])

        if not np.isfinite(ego_ttc) or not np.isfinite(other_ttc):
            continue

        if ego_ttc > 10.0:
            continue

        other_index = slot if slot < ego_index else slot + 1

        conflicts.append(
            (
                ego_ttc,
                other_ttc,
                other_index,
            )
        )

    if not conflicts:
        return MAINTAIN

    ego_ttc, other_ttc, other_index = min(
        conflicts,
        key=lambda x: x[0],
    )

    gap = other_ttc - ego_ttc

    if ego_ttc <= 1.5:
        return BRAKE

    if gap < -0.5:
        return YIELD

    if abs(gap) <= 0.5 and ego_index > other_index:
        return YIELD

    return MAINTAIN


def run(episodes: int, seed: int, background_traffic: bool):
    policies = {
        "maintain": lambda agent_id, obs: MAINTAIN,
        "safety": safest_action,
    }

    results = {}

    for name, policy in policies.items():
        env = Stage2IntersectionEnv(
            seed=seed,
            background_traffic=background_traffic,
        )

        collisions = 0
        agent_agent_collisions = 0
        background_collisions = 0
        completions = 0
        near_collisions = []
        lengths = []
        min_distances = []
        min_ttcs = []

        for episode in range(episodes):
            observations = env.reset(seed=seed + episode)

            done = False
            steps = 0
            collision = False
            background_collision = False
            episode_near = 0
            episode_min_distance = float("inf")
            episode_min_ttc = float("inf")

            while not done:
                actions = {
                    agent_id: int(
                        policy(
                            agent_id,
                            observations[agent_id],
                        )
                    )
                    for agent_id in env.agents
                }

                observations, _, terminated, truncated, infos = env.step(
                    actions
                )

                steps += 1

                global_info = infos["global"]

                episode_near = max(
                    episode_near,
                    int(global_info["near_collisions"]),
                )

                episode_min_distance = min(
                    episode_min_distance,
                    float(global_info["minimum_distance"]),
                )

                episode_min_ttc = min(
                    episode_min_ttc,
                    float(global_info["minimum_ttc"]),
                )

                if int(global_info["agent_collisions"]) > 0:
                    collision = True
                if int(global_info["traffic_collisions"]) > 0:
                    background_collision = True

                done = bool(terminated or truncated)

                if steps > env.config.max_steps:
                    raise RuntimeError("Episode exceeded configured max_steps.")

            collisions += int(collision or background_collision)
            agent_agent_collisions += int(collision)
            background_collisions += int(background_collision)
            completions += int(
                infos["global"]["all_completed"]
            )
            near_collisions.append(episode_near)
            lengths.append(steps)
            min_distances.append(episode_min_distance)
            min_ttcs.append(episode_min_ttc)

        results[name] = {
            "collision_rate": collisions / episodes,
            "agent_agent_collision_rate": agent_agent_collisions / episodes,
            "agent_traffic_collision_rate": background_collisions / episodes,
            "completion_rate": completions / episodes,
            "avg_near_collisions": float(np.mean(near_collisions)),
            "avg_episode_length": float(np.mean(lengths)),
            "mean_min_distance": float(np.mean(min_distances)),
            "worst_min_distance": float(np.min(min_distances)),
            "mean_min_ttc": float(np.mean(min_ttcs)),
            "worst_min_ttc": float(np.min(min_ttcs)),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-background", action="store_true")
    args = parser.parse_args()

    results = run(
        episodes=args.episodes,
        seed=args.seed,
        background_traffic=not args.no_background,
    )

    for name, result in results.items():
        print(name)
        for key, value in result.items():
            print(f"{key}: {value:.4f}")
        print()


if __name__ == "__main__":
    main()
