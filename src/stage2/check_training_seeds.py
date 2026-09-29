from intersection_env import Stage2IntersectionEnv


SEEDS = [42, 43, 100042, 100043]

ACTIONS = {
    "agent_0": 2,  # Maintain
    "agent_1": 2,  # Maintain
    "agent_2": 2,  # Maintain
}


for seed in SEEDS:

    print("\n" + "=" * 70)
    print(f"SEED {seed}")
    print("=" * 70)

    env = Stage2IntersectionEnv(
        seed=seed,
        background_traffic=True,
    )

    env.reset(seed=seed)

    print("\nLearning vehicles:")
    for agent_id, vehicle in env.vehicles.items():
        print(
            f"{agent_id}: "
            f"approach={vehicle.approach}, "
            f"maneuver={vehicle.maneuver}, "
            f"lane={vehicle.lane}, "
            f"speed={vehicle.speed:.3f}, "
            f"x={vehicle.x:.3f}, "
            f"y={vehicle.y:.3f}"
        )

    print("\nBackground vehicles:", len(env.background_vehicles))

    observations, rewards, terminated, truncated, infos = env.step(
        ACTIONS
    )

    global_info = infos["global"]

    print("\nAfter one safe reference action:")
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("collisions:", global_info["total_collisions"])
    print("near collisions:", global_info["near_collisions"])
    print(
        "minimum distance:",
        global_info["minimum_distance"],
    )
    print(
        "minimum TTC:",
        global_info["minimum_ttc"],
    )

    env.close() if hasattr(env, "close") else None