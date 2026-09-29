from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.envs.intersection_env import IntersectionEnv


def run_episode(
    seed,
    action,
):

    env = IntersectionEnv(
        behavior="neutral",
        seed=seed,
        max_steps=600,
    )

    obs, _ = env.reset(seed=seed)

    routes = [
        vehicle.route
        for vehicle in env.vehicles
    ]

    for _ in range(600):

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        if terminated or truncated:
            break

    return {
        "vehicles": len(env.vehicles),
        "routes": routes,
        "collision": (
            env.total_collisions > 0
        ),
        "near_collisions": (
            env.near_collisions
        ),
        "minimum_distance": (
            env.minimum_distance_seen
        ),
        "minimum_ttc": (
            env.minimum_ttc_seen
        ),
        "completed": (
            env.learning_vehicle.completed
        ),
        "final_speed": (
            env.learning_vehicle.speed
        ),
        "steps": env.step_count,
    }


print()
print("=" * 60)
print("STAGE 1 ENVIRONMENT VALIDATION")
print("=" * 60)

# =========================================================
# RANDOM TRAFFIC
# =========================================================

print()
print("1. RANDOM TRAFFIC")

episodes = []

for seed in range(10):

    result = run_episode(
        seed,
        action=2,
    )

    episodes.append(result)

    print(
        f"Episode {seed}: "
        f"vehicles={result['vehicles']}, "
        f"routes={result['routes']}, "
        f"collision={result['collision']}, "
        f"completed={result['completed']}"
    )

vehicle_counts = [
    episode["vehicles"]
    for episode in episodes
]

print(
    "Vehicle count variation:",
    len(set(vehicle_counts)) > 1,
)

# =========================================================
# ROUTES
# =========================================================

print()
print("2. ROUTE VARIATION")

all_routes = []

for episode in episodes:
    all_routes.extend(
        episode["routes"]
    )

print(
    "Routes observed:",
    set(all_routes),
)

print(
    "Straight observed:",
    "straight" in all_routes,
)

print(
    "Left observed:",
    "left" in all_routes,
)

print(
    "Right observed:",
    "right" in all_routes,
)

# =========================================================
# BRAKE
# =========================================================

print()
print("3. BRAKE TEST")

env = IntersectionEnv(
    behavior="neutral",
    seed=100,
)

obs, _ = env.reset()

speed_before = (
    env.learning_vehicle.speed
)

for _ in range(5):
    env.step(0)

speed_after = (
    env.learning_vehicle.speed
)

print(
    "Speed before:",
    round(speed_before, 3),
)

print(
    "Speed after :",
    round(speed_after, 3),
)

print(
    "Brake works:",
    speed_after < speed_before,
)

# =========================================================
# YIELD
# =========================================================

print()
print("4. YIELD TEST")

env = IntersectionEnv(
    behavior="neutral",
    seed=101,
)

obs, _ = env.reset()

speed_before = (
    env.learning_vehicle.speed
)

for _ in range(5):
    env.step(1)

speed_after = (
    env.learning_vehicle.speed
)

print(
    "Speed before:",
    round(speed_before, 3),
)

print(
    "Speed after :",
    round(speed_after, 3),
)

print(
    "Yield works:",
    (
        speed_after < speed_before
        and
        speed_after > speed_before - 1.0
    ),
)

# =========================================================
# MAINTAIN
# =========================================================

print()
print("5. MAINTAIN TEST")

env = IntersectionEnv(
    behavior="neutral",
    seed=102,
)

obs, _ = env.reset()

speed_before = (
    env.learning_vehicle.speed
)

for _ in range(5):
    env.step(2)

speed_after = (
    env.learning_vehicle.speed
)

print(
    "Speed before:",
    round(speed_before, 3),
)

print(
    "Speed after :",
    round(speed_after, 3),
)

print(
    "Maintain approximately stable:",
    abs(
        speed_after
        - speed_before
    ) < 0.1,
)

# =========================================================
# ACCELERATE
# =========================================================

print()
print("6. ACCELERATION TEST")

env = IntersectionEnv(
    behavior="neutral",
    seed=103,
)

obs, _ = env.reset()

speed_before = (
    env.learning_vehicle.speed
)

for _ in range(5):
    env.step(3)

speed_after = (
    env.learning_vehicle.speed
)

print(
    "Speed before:",
    round(speed_before, 3),
)

print(
    "Speed after :",
    round(speed_after, 3),
)

print(
    "Acceleration works:",
    speed_after > speed_before,
)

# =========================================================
# COMPLETION
# =========================================================

print()
print("7. GOAL COMPLETION")

env = IntersectionEnv(
    behavior="neutral",
    seed=104,
    max_steps=600,
)

obs, _ = env.reset()

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

print(
    "Goal completed:",
    env.learning_vehicle.completed,
)

print(
    "Steps used:",
    env.step_count,
)

# =========================================================
# SAFETY
# =========================================================

print()
print("8. SAFETY METRICS")

for seed in range(5):

    result = run_episode(
        seed=200 + seed,
        action=2,
    )

    print(
        f"Episode {seed}: "
        f"collision={result['collision']}, "
        f"near_collision="
        f"{result['near_collisions']}, "
        f"minimum_distance="
        f"{result['minimum_distance']:.3f}, "
        f"minimum_ttc="
        f"{result['minimum_ttc']:.3f}"
    )

print()
print("=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)