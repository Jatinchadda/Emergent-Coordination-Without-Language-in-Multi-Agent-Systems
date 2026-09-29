import math


def distance(vehicle_a, vehicle_b):
    return math.hypot(
        vehicle_a.x - vehicle_b.x,
        vehicle_a.y - vehicle_b.y,
    )


def check_collision(vehicle_a, vehicle_b):
    dx = abs(vehicle_a.x - vehicle_b.x)
    dy = abs(vehicle_a.y - vehicle_b.y)

    collision_x = dx < (
        vehicle_a.length + vehicle_b.length
    ) / 2

    collision_y = dy < (
        vehicle_a.width + vehicle_b.width
    ) / 2

    return collision_x and collision_y


def minimum_distance(learning_vehicle, vehicles):
    distances = []

    for vehicle in vehicles:
        if vehicle.vehicle_id == learning_vehicle.vehicle_id:
            continue

        distances.append(
            distance(learning_vehicle, vehicle)
        )

    if not distances:
        return float("inf")

    return min(distances)


def _velocity_vector(vehicle):
    if vehicle.direction == "north":
        return 0.0, vehicle.speed

    if vehicle.direction == "south":
        return 0.0, -vehicle.speed

    if vehicle.direction == "east":
        return vehicle.speed, 0.0

    return -vehicle.speed, 0.0


def time_to_collision(vehicle_a, vehicle_b):
    dx = vehicle_b.x - vehicle_a.x
    dy = vehicle_b.y - vehicle_a.y

    distance_now = math.hypot(dx, dy)

    if distance_now <= 0:
        return 0.0

    ax, ay = _velocity_vector(vehicle_a)
    bx, by = _velocity_vector(vehicle_b)

    relative_vx = ax - bx
    relative_vy = ay - by

    closing_speed = (
        relative_vx * dx
        + relative_vy * dy
    ) / distance_now

    if closing_speed <= 0:
        return float("inf")

    return distance_now / closing_speed


def minimum_ttc(learning_vehicle, vehicles):
    values = []

    for vehicle in vehicles:
        if vehicle.vehicle_id == learning_vehicle.vehicle_id:
            continue

        ttc = time_to_collision(
            learning_vehicle,
            vehicle,
        )

        if math.isfinite(ttc):
            values.append(ttc)

    if not values:
        return float("inf")

    return min(values)


def check_near_collision(
    learning_vehicle,
    vehicles,
    threshold=6.0,
):
    return (
        minimum_distance(
            learning_vehicle,
            vehicles,
        )
        < threshold
    )