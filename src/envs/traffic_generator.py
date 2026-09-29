import random

from .vehicle import Vehicle


DIRECTIONS = [
    "north",
    "south",
    "east",
    "west",
]

ROUTES = [
    "straight",
    "left",
    "right",
]


class TrafficGenerator:

    def __init__(
        self,
        max_vehicles=8,
        spawn_distance=80.0,
        min_speed=3.0,
        max_speed=9.0,
        seed=None,
    ):
        self.max_vehicles = max_vehicles
        self.spawn_distance = spawn_distance
        self.min_speed = min_speed
        self.max_speed = max_speed

        self.rng = random.Random(seed)

    # =====================================================
    # RANDOMIZATION
    # =====================================================

    def random_direction(self):
        return self.rng.choice(
            DIRECTIONS
        )

    def random_route(self):
        return self.rng.choice(
            ROUTES
        )

    def random_lane(self):
        return self.rng.randint(
            0,
            1,
        )

    # =====================================================
    # SPAWN
    # =====================================================

    def spawn_position(
        self,
        direction,
        lane,
    ):

        offset = (
            3.5
            if lane == 0
            else 5.5
        )

        if direction == "north":

            return (
                -offset,
                -self.spawn_distance,
            )

        if direction == "south":

            return (
                offset,
                self.spawn_distance,
            )

        if direction == "east":

            return (
                -self.spawn_distance,
                offset,
            )

        if direction == "west":

            return (
                self.spawn_distance,
                -offset,
            )

        raise ValueError(
            f"Unknown direction: {direction}"
        )

    # =====================================================
    # VEHICLE
    # =====================================================

    def create_scripted_vehicle(
        self,
        vehicle_id,
    ):

        direction = self.random_direction()
        route = self.random_route()
        lane = self.random_lane()

        x, y = self.spawn_position(
            direction,
            lane,
        )

        speed = self.rng.uniform(
            self.min_speed,
            self.max_speed,
        )

        return Vehicle(
            vehicle_id=vehicle_id,
            x=x,
            y=y,
            speed=speed,
            direction=direction,
            lane=lane,
            route=route,
            is_learning=False,
        )

    # =====================================================
    # GENERATE
    # =====================================================

    def generate(
        self,
        learning_vehicle,
    ):

        vehicles = [
            learning_vehicle
        ]

        number_of_others = self.rng.randint(
            2,
            self.max_vehicles,
        )

        for i in range(
            number_of_others
        ):

            vehicle = (
                self.create_scripted_vehicle(
                    vehicle_id=f"traffic_{i}"
                )
            )

            if (
                vehicle.distance_to(
                    learning_vehicle
                )
                < 15.0
            ):
                continue

            too_close = False

            for existing in vehicles:

                if (
                    vehicle.distance_to(
                        existing
                    )
                    < 12.0
                ):
                    too_close = True
                    break

            if too_close:
                continue

            vehicles.append(vehicle)

        return vehicles