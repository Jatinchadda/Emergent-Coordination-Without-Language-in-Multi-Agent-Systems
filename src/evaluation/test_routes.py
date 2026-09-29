from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.envs.intersection_env import IntersectionEnv


ROUTES = [
    "straight",
    "left",
    "right",
]


print()
print("ROUTE TEST")
print()

for route in ROUTES:

    env = IntersectionEnv(
        behavior="neutral",
        seed=42,
        max_steps=600,
    )

    obs, _ = env.reset()

    env.learning_vehicle.route = route

    start_position = (
        env.learning_vehicle.position()
    )

    for _ in range(600):

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(2)

        if terminated or truncated:
            break

    end_position = (
        env.learning_vehicle.position()
    )

    print("-" * 30)

    print(
        f"Route: {route}"
    )

    print(
        "Start position:",
        start_position,
    )

    print(
        "End position:",
        end_position,
    )

    print(
        "Steps:",
        env.step_count,
    )

    print(
        "Completed:",
        env.learning_vehicle.completed,
    )

    print(
        "Collision:",
        env.total_collisions > 0,
    )

print()
print("Route tests completed.")