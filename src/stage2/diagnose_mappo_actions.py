from intersection_env import Stage2IntersectionEnv


TEST_ACTIONS = [
    {"agent_0": 0, "agent_1": 0, "agent_2": 3},
    {"agent_0": 0, "agent_1": 1, "agent_2": 2},
    {"agent_0": 0, "agent_1": 3, "agent_2": 1},
]


def run_case(actions):
    env = Stage2IntersectionEnv(seed=42, background_traffic=True)

    env.reset()

    print("\n" + "=" * 70)
    print(f"ACTIONS: {actions}")

    print("Initial vehicles:")
    for agent_id, vehicle in env.vehicles.items():
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

    obs, rewards, terminated, truncated, infos = env.step(actions)

    print("\nAfter one step:")
    print(f"  rewards: {rewards}")
    print(f"  terminated: {terminated}")
    print(f"  truncated: {truncated}")

    global_info = infos["global"]

    print("\nGlobal:")
    print(f"  step: {global_info['step']}")
    print(f"  total_collisions: {global_info['total_collisions']}")
    print(f"  agent_collisions: {global_info['agent_collisions']}")
    print(f"  traffic_collisions: {global_info['traffic_collisions']}")
    print(f"  near_collisions: {global_info['near_collisions']}")
    print(f"  minimum_distance: {global_info['minimum_distance']:.3f}")
    print(f"  minimum_ttc: {global_info['minimum_ttc']:.3f}")
    print(f"  all_completed: {global_info['all_completed']}")

    print("\nAgent states:")
    for agent_id, vehicle in env.vehicles.items():
        print(
            f"  {agent_id}: "
            f"speed={vehicle.speed:.3f}, "
            f"acceleration={vehicle.acceleration:.3f}, "
            f"x={vehicle.x:.3f}, "
            f"y={vehicle.y:.3f}, "
            f"collided={vehicle.collided}"
        )

    env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
    for actions in TEST_ACTIONS:
        run_case(actions)