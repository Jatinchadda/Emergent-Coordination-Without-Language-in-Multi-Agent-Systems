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


SCENARIOS = {
    "cross": [
        {
            "agent_id": "agent_0",
            "approach": "north",
            "maneuver": "straight",
            "lane": 0,
            "speed": 7.0,
        },
        {
            "agent_id": "agent_1",
            "approach": "west",
            "maneuver": "straight",
            "lane": 0,
            "speed": 7.0,
        },
        {
            "agent_id": "agent_2",
            "approach": "south",
            "maneuver": "right",
            "lane": 0,
            "speed": 7.0,
        },
    ],
    "opposite": [
        {
            "agent_id": "agent_0",
            "approach": "north",
            "maneuver": "straight",
            "lane": 0,
            "speed": 7.0,
        },
        {
            "agent_id": "agent_1",
            "approach": "south",
            "maneuver": "straight",
            "lane": 0,
            "speed": 7.0,
        },
        {
            "agent_id": "agent_2",
            "approach": "east",
            "maneuver": "left",
            "lane": 0,
            "speed": 7.0,
        },
    ],
    "three_way": [
        {
            "agent_id": "agent_0",
            "approach": "north",
            "maneuver": "straight",
            "lane": 0,
            "speed": 7.0,
        },
        {
            "agent_id": "agent_1",
            "approach": "west",
            "maneuver": "straight",
            "lane": 0,
            "speed": 7.0,
        },
        {
            "agent_id": "agent_2",
            "approach": "south",
            "maneuver": "left",
            "lane": 0,
            "speed": 7.0,
        },
    ],
}


def ttc_seconds(value: float) -> float:
    if not np.isfinite(value):
        return float("inf")
    return float(np.clip(value, 0.0, 1.0) * 10.0)


def action_for_agent(agent_id: str, observation: np.ndarray) -> int:
    ego_index = int(agent_id.rsplit("_", 1)[1])

    blocks = (
        observation[11:19],
        observation[19:27],
    )

    conflicts = []

    for slot, block in enumerate(blocks):
        if block[7] < 0.5:
            continue

        ego_ttc = ttc_seconds(block[4])
        other_ttc = ttc_seconds(block[5])

        if not np.isfinite(ego_ttc) or not np.isfinite(other_ttc):
            continue

        if ego_ttc > 8.0:
            continue

        other_index = slot if slot < ego_index else slot + 1
        conflicts.append((ego_ttc, other_ttc, other_index))

    if not conflicts:
        return MAINTAIN

    ego_ttc, other_ttc, other_index = min(conflicts, key=lambda x: x[0])
    gap = other_ttc - ego_ttc

    if ego_ttc <= 1.0:
        return BRAKE

    if gap > 1.0:
        return YIELD

    if abs(gap) <= 1.0 and ego_index > other_index:
        return YIELD

    return MAINTAIN


def run_scenario(name: str, scenario: dict, episodes: int, seed: int):
    env = Stage2IntersectionEnv(
        seed=seed,
        background_traffic=False,
    )

    collisions = 0
    completions = 0
    min_distances = []
    min_ttcs = []
    lengths = []

    for episode in range(episodes):
        observations = env.reset(
            seed=seed + episode,
            scenario=scenario,
        )

        done = False
        steps = 0
        episode_collision = False
        episode_min_distance = float("inf")
        episode_min_ttc = float("inf")

        while not done:
            actions = {
                agent_id: action_for_agent(
                    agent_id,
                    observations[agent_id],
                )
                for agent_id in env.agents
            }

            observations, _, terminated, truncated, infos = env.step(actions)

            steps += 1

            global_info = infos["global"]

            episode_min_distance = min(
                episode_min_distance,
                float(global_info["minimum_distance"]),
            )
            episode_min_ttc = min(
                episode_min_ttc,
                float(global_info["minimum_ttc"]),
            )

            if int(global_info["total_collisions"]) > 0:
                episode_collision = True

            done = bool(terminated or truncated)

            if steps > env.config.max_steps:
                raise RuntimeError("Episode exceeded configured max_steps.")

        collisions += int(episode_collision)
        completions += int(infos["global"]["all_completed"])
        min_distances.append(episode_min_distance)
        min_ttcs.append(episode_min_ttc)
        lengths.append(steps)

    return {
        "collision_rate": collisions / episodes,
        "completion_rate": completions / episodes,
        "mean_min_distance": float(np.mean(min_distances)),
        "worst_min_distance": float(np.min(min_distances)),
        "mean_min_ttc": float(np.mean(min_ttcs)),
        "worst_min_ttc": float(np.min(min_ttcs)),
        "avg_episode_length": float(np.mean(lengths)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for name, scenario in SCENARIOS.items():
        result = run_scenario(
            name=name,
            scenario=scenario,
            episodes=args.episodes,
            seed=args.seed,
        )

        print(name)
        for key, value in result.items():
            print(f"{key}: {value:.4f}")
        print()


if __name__ == "__main__":
    main()
