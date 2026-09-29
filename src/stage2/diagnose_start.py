from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STAGE2 = ROOT / "src" / "stage2"

sys.path.insert(0, str(STAGE2))

from intersection_env import (
    Stage2IntersectionEnv,
    Stage2Config,
)


def run_case(name, actions):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    env = Stage2IntersectionEnv(
        seed=42,
        config=Stage2Config(),
        background_traffic=True,
    )

    observations = env.reset(seed=42)

    print("Initial learning vehicles:")

    for agent_id in env.agents:

        vehicle = env.vehicles[agent_id]

        print(
            f"  {agent_id}: "
            f"behavior={vehicle.behavior}, "
            f"approach={vehicle.approach}, "
            f"maneuver={vehicle.maneuver}, "
            f"lane={vehicle.lane}, "
            f"speed={vehicle.speed:.3f}, "
            f"x={vehicle.x:.3f}, "
            f"y={vehicle.y:.3f}"
        )

    print()

    print(
        f"Background vehicles: "
        f"{len(env.background_vehicles)}"
    )

    print()

    print("Actions:")
    print(actions)

    (
        next_observations,
        rewards,
        terminated,
        truncated,
        infos,
    ) = env.step(actions)

    print()
    print("Result:")

    print(
        f"  rewards:    {rewards}"
    )

    print(
        f"  terminated: {terminated}"
    )

    print(
        f"  truncated:  {truncated}"
    )

    print()
    print("Global metrics:")

    global_info = infos["global"]

    for key, value in global_info.items():

        print(
            f"  {key}: {value}"
        )


if __name__ == "__main__":

    run_case(
        "ALL MAINTAIN",
        {
            "agent_0": 2,
            "agent_1": 2,
            "agent_2": 2,
        },
    )

    run_case(
        "ALL BRAKE",
        {
            "agent_0": 0,
            "agent_1": 0,
            "agent_2": 0,
        },
    )

    run_case(
        "ALL YIELD",
        {
            "agent_0": 1,
            "agent_1": 1,
            "agent_2": 1,
        },
    )

    run_case(
        "ALL ACCELERATE",
        {
            "agent_0": 3,
            "agent_1": 3,
            "agent_2": 3,
        },
    )