"""
Stage 2 multi-agent environment validation.

Run from project root:

    python -m src.evaluation.test_multi_agent_environment
"""

from collections import Counter

import numpy as np

from src.envs.multi_agent_intersection_env import (
    MultiAgentIntersectionEnv,
)


def run():

    env = MultiAgentIntersectionEnv(
        seed=2026,
        max_steps=300,
    )

    # =========================================================
    # RESET
    # =========================================================

    observations, infos = env.reset(
        seed=2026
    )

    assert set(
        observations
    ) == set(
        env.agent_ids
    )

    assert all(
        observation.shape
        == (env.observation_dim,)
        for observation in observations.values()
    )

    assert all(
        np.isfinite(
            observation
        ).all()
        for observation in observations.values()
    )

    print("=" * 70)
    print("STAGE 2 MULTI-AGENT ENVIRONMENT")
    print("=" * 70)

    print(
        f"Agents: {env.agent_ids}"
    )

    print(
        f"Behaviors: {env.behaviors}"
    )

    print(
        f"Observation dimension: "
        f"{env.observation_dim}"
    )

    print(
        f"Action dimension: "
        f"{env.action_dim}"
    )

    print()

    # =========================================================
    # INITIAL SCENARIO
    # =========================================================

    print("Initial scenario:")

    for agent_id in env.agent_ids:

        vehicle = (
            env.learning_vehicles[
                agent_id
            ]
        )

        print(
            f"  {agent_id}: "
            f"{env.agent_behavior[agent_id]} | "
            f"direction={vehicle.direction} | "
            f"route={vehicle.route} | "
            f"lane={vehicle.lane} | "
            f"speed={vehicle.speed:.2f}"
        )

    print()

    # =========================================================
    # ACTION COVERAGE
    # =========================================================

    action_counts = {
        agent_id: Counter()
        for agent_id in env.agent_ids
    }

    steps = 0

    # =========================================================
    # SIMULTANEOUS MULTI-AGENT STEPPING
    # =========================================================

    for step in range(300):

        actions = {
            "agent_0":
                step % 4,

            "agent_1":
                (step + 1) % 4,

            "agent_2":
                (step + 2) % 4,
        }

        for agent_id, action in actions.items():

            action_counts[
                agent_id
            ][action] += 1

        (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = env.step(
            actions
        )

        steps += 1

        # -----------------------------------------------------
        # API checks
        # -----------------------------------------------------

        assert set(
            observations
        ) == set(
            env.agent_ids
        )

        assert set(
            rewards
        ) == set(
            env.agent_ids
        )

        assert set(
            terminated
        ) == set(
            env.agent_ids
        )

        assert set(
            truncated
        ) == set(
            env.agent_ids
        )

        # -----------------------------------------------------
        # Observation checks
        # -----------------------------------------------------

        for agent_id in env.agent_ids:

            assert (
                observations[
                    agent_id
                ].shape
                == (
                    env.observation_dim,
                )
            )

            assert np.isfinite(
                observations[
                    agent_id
                ]
            ).all()

        # -----------------------------------------------------
        # End condition
        # -----------------------------------------------------

        if (
            all(
                terminated.values()
            )
            or
            all(
                truncated.values()
            )
        ):

            break

    # =========================================================
    # SEED REPRODUCIBILITY
    # =========================================================

    first_obs, _ = env.reset(
        seed=777
    )

    first_state = [
        (
            env.learning_vehicles[
                agent_id
            ].x,

            env.learning_vehicles[
                agent_id
            ].y,

            env.learning_vehicles[
                agent_id
            ].direction,

            env.learning_vehicles[
                agent_id
            ].route,
        )
        for agent_id in env.agent_ids
    ]

    second_obs, _ = env.reset(
        seed=777
    )

    second_state = [
        (
            env.learning_vehicles[
                agent_id
            ].x,

            env.learning_vehicles[
                agent_id
            ].y,

            env.learning_vehicles[
                agent_id
            ].direction,

            env.learning_vehicles[
                agent_id
            ].route,
        )
        for agent_id in env.agent_ids
    ]

    assert (
        first_state
        == second_state
    )

    for agent_id in env.agent_ids:

        assert np.array_equal(
            first_obs[
                agent_id
            ],
            second_obs[
                agent_id
            ],
        )

    # =========================================================
    # RESULTS
    # =========================================================

    print(
        "Simultaneous action stepping: PASS"
    )

    print(
        "Finite observations: PASS"
    )

    print(
        "Seed reproducibility: PASS"
    )

    print(
        f"Steps executed: {steps}"
    )

    print()

    print("Action coverage:")

    for agent_id in env.agent_ids:

        print(
            f"  {agent_id}: "
            f"{dict(sorted(action_counts[agent_id].items()))}"
        )

    print()

    print("Metrics:")

    print(
        f"  total_collisions = "
        f"{env.total_collisions}"
    )

    print(
        f"  agent_collisions = "
        f"{env.agent_collisions}"
    )

    print(
        f"  traffic_collisions = "
        f"{env.traffic_collisions}"
    )

    print(
        f"  near_collisions = "
        f"{env.near_collisions}"
    )

    print(
        f"  minimum_distance_seen = "
        f"{env.minimum_distance_seen:.3f}"
    )

    print(
        f"  minimum_ttc_seen = "
        f"{env.minimum_ttc_seen:.3f}"
    )

    print()

    print("=" * 70)
    print(
        "STAGE 2 ENVIRONMENT TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    run()