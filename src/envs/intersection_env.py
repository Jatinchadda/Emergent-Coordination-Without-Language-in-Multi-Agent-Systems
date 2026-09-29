import math
import random

import numpy as np

from .collision import (
    check_collision,
    minimum_distance,
    minimum_ttc,
)

from .traffic_generator import TrafficGenerator
from .vehicle import Vehicle

from src.agents.behavior_profiles import (
    get_behavior_profile,
)


class IntersectionEnv:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        behavior="aggressive",
        seed=42,
        max_steps=600,
    ):

        if behavior not in {
            "aggressive",
            "neutral",
            "conservative",
        }:
            raise ValueError(
                f"Unknown behavior: {behavior}"
            )

        self.behavior = behavior
        self.max_steps = max_steps

        self.rng = random.Random(seed)

        # =====================================================
        # SIMULATION
        # =====================================================

        self.dt = 0.1

        # =====================================================
        # SHARED VEHICLE PHYSICS
        # =====================================================

        self.max_speed = 12.0

        # Same physical capability for all behaviors.
        self.max_acceleration = 2.0
        self.yield_deceleration = -1.0
        self.max_braking = -2.0

        # =====================================================
        # ROAD / INTERSECTION
        # =====================================================

        self.spawn_distance = 80.0
        self.goal_distance = 90.0

        # Central conflict zone.
        self.conflict_radius = 14.0
        self.conflict_zone_length = 28.0

        # =====================================================
        # TRAFFIC
        # =====================================================

        self.traffic_generator = TrafficGenerator(
            max_vehicles=8,
            spawn_distance=self.spawn_distance,
            min_speed=3.0,
            max_speed=9.0,
            seed=seed,
        )

        # =====================================================
        # BEHAVIOR PROFILE
        # =====================================================

        self.behavior_profile = get_behavior_profile(
            behavior
        )

        # =====================================================
        # PPO INTERFACE
        # =====================================================

        self.observation_dim = 40

        # 0 = Brake
        # 1 = Yield
        # 2 = Maintain
        # 3 = Accelerate
        self.action_dim = 4

        # =====================================================
        # STATE
        # =====================================================

        self.vehicles = []
        self.learning_vehicle = None

        self.step_count = 0

        # =====================================================
        # REWARD STATE
        # =====================================================

        # Risk at the previous timestep.
        # Used for risk-change / recovery reward.
        self.previous_conflict_risk = 0.0

        self.previous_conflict_ttc = float("inf")

        # Remaining distance to route-specific goal.
        self.previous_remaining_distance = None

        # =====================================================
        # METRICS
        # =====================================================

        self.total_collisions = 0
        self.near_collisions = 0

        self.minimum_distance_seen = float("inf")
        self.minimum_ttc_seen = float("inf")

    # =========================================================
    # RESET
    # =========================================================

    def reset(self, seed=None):

        if seed is not None:
            self.rng.seed(seed)
            self.traffic_generator.rng.seed(seed)

        self.step_count = 0

        self.total_collisions = 0
        self.near_collisions = 0

        self.minimum_distance_seen = float("inf")
        self.minimum_ttc_seen = float("inf")

        self.previous_conflict_risk = 0.0
        self.previous_conflict_ttc = float("inf")

        self.previous_remaining_distance = None

        self.vehicles = []

        # -----------------------------------------------------
        # Learning vehicle scenario
        # -----------------------------------------------------

        direction = self.rng.choice(
            [
                "north",
                "south",
                "east",
                "west",
            ]
        )

        lane = self.rng.randint(0, 1)

        route = self.rng.choice(
            [
                "straight",
                "left",
                "right",
            ]
        )

        x, y = self._learning_spawn(
            direction,
            lane,
        )

        speed = self.rng.uniform(
            3.0,
            7.0,
        )

        self.learning_vehicle = Vehicle(
            vehicle_id="learning",
            x=x,
            y=y,
            speed=speed,
            direction=direction,
            lane=lane,
            route=route,
            is_learning=True,
        )

        # -----------------------------------------------------
        # Scripted traffic
        # -----------------------------------------------------

        self.vehicles = (
            self.traffic_generator.generate(
                self.learning_vehicle
            )
        )

        # -----------------------------------------------------
        # Initialize reward state
        # -----------------------------------------------------

        self.previous_remaining_distance = (
            self._remaining_goal_distance()
        )

        conflict = self._get_conflict_risk()

        self.previous_conflict_risk = (
            conflict["risk"]
        )

        self.previous_conflict_ttc = (
            conflict["ttc"]
        )

        return (
            self._get_observation(),
            {},
        )

    # =========================================================
    # LEARNING VEHICLE SPAWN
    # =========================================================

    def _learning_spawn(
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

    # =========================================================
    # STEP
    # =========================================================

    def step(self, action):

        action = int(action)

        if action not in (
            0,
            1,
            2,
            3,
        ):
            raise ValueError(
                f"Invalid action: {action}. "
                "Expected 0=Brake, 1=Yield, "
                "2=Maintain, 3=Accelerate."
            )

        self.step_count += 1

        # -----------------------------------------------------
        # State before action
        # -----------------------------------------------------

        previous_risk = (
            self.previous_conflict_risk
        )

        previous_distance = (
            self.previous_remaining_distance
        )

        # -----------------------------------------------------
        # Convert action to acceleration
        # -----------------------------------------------------

        acceleration = (
            self._action_to_acceleration(
                action
            )
        )

        # -----------------------------------------------------
        # Move learning vehicle
        # -----------------------------------------------------

        self.learning_vehicle.update(
            acceleration,
            self.dt,
        )

        self.learning_vehicle.speed_limit_check(
            self.max_speed
        )

        # -----------------------------------------------------
        # Move scripted traffic
        # -----------------------------------------------------

        self._update_scripted_traffic()

        # -----------------------------------------------------
        # Collision
        # -----------------------------------------------------

        collision = (
            self._check_collisions()
        )

        # -----------------------------------------------------
        # Safety metrics
        # -----------------------------------------------------

        min_distance = minimum_distance(
            self.learning_vehicle,
            self.vehicles,
        )

        current_ttc = minimum_ttc(
            self.learning_vehicle,
            self.vehicles,
        )

        if np.isfinite(min_distance):

            self.minimum_distance_seen = min(
                self.minimum_distance_seen,
                min_distance,
            )

        if np.isfinite(current_ttc):

            self.minimum_ttc_seen = min(
                self.minimum_ttc_seen,
                current_ttc,
            )

        near_collision = (
            np.isfinite(min_distance)
            and min_distance < 6.0
        )

        if near_collision:
            self.near_collisions += 1

        # -----------------------------------------------------
        # Conflict risk after movement
        # -----------------------------------------------------

        conflict = (
            self._get_conflict_risk()
        )

        current_risk = (
            conflict["risk"]
        )

        current_conflict_ttc = (
            conflict["ttc"]
        )

        if not np.isfinite(
            current_conflict_ttc
        ):
            current_conflict_ttc = float("inf")

        # Risk change:
        # positive = becoming more dangerous
        # negative = situation is clearing
        risk_change = (
            current_risk
            - previous_risk
        )

        # -----------------------------------------------------
        # Actual progress toward route goal
        # -----------------------------------------------------

        current_distance = (
            self._remaining_goal_distance()
        )

        if previous_distance is None:

            progress_signal = 0.0

        else:

            progress_delta = (
                previous_distance
                - current_distance
            )

            max_step_distance = (
                self.max_speed
                * self.dt
            )

            progress_signal = float(
                np.clip(
                    progress_delta
                    / max(
                        max_step_distance,
                        1e-6,
                    ),
                    -1.0,
                    1.0,
                )
            )

        # -----------------------------------------------------
        # Completion
        # -----------------------------------------------------

        completed = (
            self._reached_goal()
        )

        if completed:
            self.learning_vehicle.completed = True

        # -----------------------------------------------------
        # Reward
        # -----------------------------------------------------

        reward = self._calculate_reward(
            action=action,
            collision=collision,
            completed=completed,
            progress_signal=progress_signal,
            current_risk=current_risk,
            previous_risk=previous_risk,
            risk_change=risk_change,
            near_collision=near_collision,
        )

        # -----------------------------------------------------
        # Termination
        # -----------------------------------------------------

        terminated = False
        truncated = False

        if collision:
            terminated = True

        if completed:
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        # -----------------------------------------------------
        # Store state for next timestep
        # -----------------------------------------------------

        self.previous_remaining_distance = (
            current_distance
        )

        self.previous_conflict_risk = (
            current_risk
        )

        self.previous_conflict_ttc = (
            current_conflict_ttc
        )

        # -----------------------------------------------------
        # Info
        # -----------------------------------------------------

        info = {

            "collision": collision,

            "near_collision": near_collision,

            "completed": completed,

            "minimum_distance": min_distance,

            "minimum_ttc": current_ttc,

            "conflict_risk": current_risk,

            "previous_conflict_risk": previous_risk,

            "risk_change": risk_change,

            "conflict_ttc": current_conflict_ttc,

            "conflict_gap": conflict["gap"],

            "progress_signal": progress_signal,

            "remaining_goal_distance": current_distance,

            "speed": self.learning_vehicle.speed,

            "acceleration": (
                self.learning_vehicle.acceleration
            ),

            "route": self.learning_vehicle.route,

            "direction": self.learning_vehicle.direction,

            "behavior": self.behavior,

            "action": action,

            "step": self.step_count,
        }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    # =========================================================
    # ACTION MAPPING
    # =========================================================

    def _action_to_acceleration(
        self,
        action,
    ):

        if action == 0:
            # Brake
            return self.max_braking

        if action == 1:
            # Yield
            return self.yield_deceleration

        if action == 2:
            # Maintain
            return 0.0

        if action == 3:
            # Accelerate
            return self.max_acceleration

        raise ValueError(
            f"Invalid action: {action}"
        )

    # =========================================================
    # SCRIPTED TRAFFIC
    # =========================================================

    def _update_scripted_traffic(self):

        for vehicle in self.vehicles:

            if vehicle.is_learning:
                continue

            variation = self.rng.uniform(
                -0.5,
                0.5,
            )

            target_speed = np.clip(
                vehicle.speed + variation,
                2.0,
                9.0,
            )

            if vehicle.speed < target_speed:

                acceleration = 1.0

            else:

                acceleration = -0.5

            # Occasional natural braking.
            if self.rng.random() < 0.015:
                acceleration = -2.0

            vehicle.update(
                acceleration,
                self.dt,
            )

            vehicle.speed_limit_check(
                9.0
            )

    # =========================================================
    # COLLISION
    # =========================================================

    def _check_collisions(self):

        collision = False

        for vehicle in self.vehicles:

            if vehicle.is_learning:
                continue

            if check_collision(
                self.learning_vehicle,
                vehicle,
            ):

                self.learning_vehicle.collided = True
                vehicle.collided = True

                collision = True

                self.total_collisions += 1

        return collision

    # =========================================================
    # ROUTE HELPERS
    # =========================================================

    @staticmethod
    def _exit_direction(vehicle):

        if vehicle.route == "straight":
            return vehicle.direction

        exits = {

            "north": {
                "left": "west",
                "right": "east",
            },

            "south": {
                "left": "east",
                "right": "west",
            },

            "east": {
                "left": "north",
                "right": "south",
            },

            "west": {
                "left": "south",
                "right": "north",
            },
        }

        return exits[
            vehicle.direction
        ][vehicle.route]

    @staticmethod
    def _routes_conflict(
        ego,
        other,
    ):

        # Same approach is not an intersection crossing conflict.
        if ego.direction == other.direction:
            return False

        perpendicular_pairs = {

            ("north", "east"),
            ("north", "west"),
            ("south", "east"),
            ("south", "west"),
        }

        pair = (
            ego.direction,
            other.direction,
        )

        reverse_pair = (
            other.direction,
            ego.direction,
        )

        if (
            pair in perpendicular_pairs
            or
            reverse_pair in perpendicular_pairs
        ):
            return True

        # Check whether one vehicle exits through
        # the other vehicle's approach.
        ego_exit = (
            IntersectionEnv._exit_direction(
                ego
            )
        )

        other_exit = (
            IntersectionEnv._exit_direction(
                other
            )
        )

        if (
            ego_exit == other.direction
            or
            other_exit == ego.direction
        ):
            return True

        return False

    # =========================================================
    # CONFLICT-ZONE GEOMETRY
    # =========================================================

    def _is_beyond_exit(self, vehicle):

        exit_direction = (
            self._exit_direction(vehicle)
        )

        if exit_direction == "north":
            return vehicle.y > self.conflict_radius

        if exit_direction == "south":
            return vehicle.y < -self.conflict_radius

        if exit_direction == "east":
            return vehicle.x > self.conflict_radius

        return vehicle.x < -self.conflict_radius

    def _eta_to_conflict_zone(self, vehicle):

        # Vehicle has already cleared the conflict zone.
        if self._is_beyond_exit(vehicle):

            return (
                float("inf"),
                0.0,
            )

        center_distance = math.hypot(
            vehicle.x,
            vehicle.y,
        )

        # Already inside conflict zone.
        if (
            getattr(
                vehicle,
                "in_intersection",
                False,
            )
            or
            center_distance <= self.conflict_radius
        ):

            eta = 0.0

        else:

            distance_to_zone = (
                center_distance
                - self.conflict_radius
            )

            speed = max(
                vehicle.speed,
                0.5,
            )

            eta = (
                distance_to_zone
                / speed
            )

        speed = max(
            vehicle.speed,
            0.5,
        )

        occupancy = (
            self.conflict_zone_length
            / speed
        )

        return (
            float(eta),
            float(occupancy),
        )

    # =========================================================
    # RELATIVE TTC
    # =========================================================

    @staticmethod
    def _heading_from_direction(
        direction,
    ):

        values = {

            "east": 0.0,

            "north": np.pi / 2.0,

            "west": np.pi,

            "south": -np.pi / 2.0,
        }

        return values.get(
            direction,
            0.0,
        )

    def _relative_ttc(
        self,
        ego,
        other,
    ):
        """
        Estimate line-of-sight closing TTC.

        Only positive closing speed produces a finite TTC.
        """

        dx = other.x - ego.x
        dy = other.y - ego.y

        distance = math.hypot(
            dx,
            dy,
        )

        if distance <= 1e-6:
            return 0.0

        ego_heading = getattr(
            ego,
            "heading",
            self._heading_from_direction(
                ego.direction
            ),
        )

        other_heading = getattr(
            other,
            "heading",
            self._heading_from_direction(
                other.direction
            ),
        )

        ego_vx = (
            ego.speed
            * math.cos(ego_heading)
        )

        ego_vy = (
            ego.speed
            * math.sin(ego_heading)
        )

        other_vx = (
            other.speed
            * math.cos(other_heading)
        )

        other_vy = (
            other.speed
            * math.sin(other_heading)
        )

        relative_vx = (
            other_vx - ego_vx
        )

        relative_vy = (
            other_vy - ego_vy
        )

        closing_speed = -(
            dx * relative_vx
            + dy * relative_vy
        ) / distance

        if closing_speed <= 1e-6:
            return float("inf")

        return float(
            distance
            / closing_speed
        )

    # =========================================================
    # CONFLICT RISK
    # =========================================================

    def _get_conflict_risk(self):

        ego = self.learning_vehicle

        best = {

            "vehicle": None,

            "risk": 0.0,

            "ttc": float("inf"),

            "gap": float("inf"),
        }

        ego_eta, ego_occupancy = (
            self._eta_to_conflict_zone(
                ego
            )
        )

        for other in self.vehicles:

            if other.is_learning:
                continue

            if not self._routes_conflict(
                ego,
                other,
            ):
                continue

            other_eta, other_occupancy = (
                self._eta_to_conflict_zone(
                    other
                )
            )

            # A vehicle that has already cleared the
            # intersection does not create future conflict.
            if not np.isfinite(other_eta):
                continue

            if not np.isfinite(ego_eta):
                continue

            # -------------------------------------------------
            # Arrival-window overlap
            # -------------------------------------------------

            ego_start = ego_eta

            ego_end = (
                ego_eta
                + ego_occupancy
            )

            other_start = other_eta

            other_end = (
                other_eta
                + other_occupancy
            )

            overlap = max(
                0.0,
                min(
                    ego_end,
                    other_end,
                )
                - max(
                    ego_start,
                    other_start,
                ),
            )

            minimum_occupancy = max(
                min(
                    ego_occupancy,
                    other_occupancy,
                ),
                1e-6,
            )

            timing_risk = np.clip(
                overlap
                / minimum_occupancy,
                0.0,
                1.0,
            )

            # -------------------------------------------------
            # Relative TTC
            # -------------------------------------------------

            ttc = self._relative_ttc(
                ego,
                other,
            )

            if np.isfinite(ttc):

                ttc_risk = np.exp(
                    -max(ttc, 0.0)
                    / 2.0
                )

            else:

                ttc_risk = 0.0

            # -------------------------------------------------
            # Proximity
            # -------------------------------------------------

            distance = ego.distance_to(
                other
            )

            proximity_risk = np.exp(
                -max(distance, 0.0)
                / 8.0
            )

            # -------------------------------------------------
            # Both vehicles in intersection
            # -------------------------------------------------

            ego_inside = (
                abs(ego.x)
                <= self.conflict_radius
                and
                abs(ego.y)
                <= self.conflict_radius
            )

            other_inside = (
                abs(other.x)
                <= self.conflict_radius
                and
                abs(other.y)
                <= self.conflict_radius
            )

            if (
                ego_inside
                and other_inside
            ):

                timing_risk = max(
                    timing_risk,
                    0.80,
                )

                proximity_risk = max(
                    proximity_risk,
                    0.60,
                )

            # -------------------------------------------------
            # Combined risk
            # -------------------------------------------------

            risk = (
                0.55 * timing_risk
                +
                0.25 * ttc_risk
                +
                0.20 * proximity_risk
            )

            risk = float(
                np.clip(
                    risk,
                    0.0,
                    1.0,
                )
            )

            if risk > best["risk"]:

                best = {

                    "vehicle": other,

                    "risk": risk,

                    "ttc": float(ttc),

                    # Gap here means ETA gap.
                    "gap": float(
                        abs(
                            ego_eta
                            - other_eta
                        )
                    ),
                }

        return best

    # =========================================================
    # GOAL
    # =========================================================

    def _goal_point(self):

        v = self.learning_vehicle

        exit_direction = (
            self._exit_direction(v)
        )

        if exit_direction == "north":

            return (
                0.0,
                self.goal_distance,
            )

        if exit_direction == "south":

            return (
                0.0,
                -self.goal_distance,
            )

        if exit_direction == "east":

            return (
                self.goal_distance,
                0.0,
            )

        return (
            -self.goal_distance,
            0.0,
        )

    def _remaining_goal_distance(self):

        v = self.learning_vehicle

        goal_x, goal_y = (
            self._goal_point()
        )

        return float(
            math.hypot(
                goal_x - v.x,
                goal_y - v.y,
            )
        )

    def _reached_goal(self):

        return bool(
            self.learning_vehicle.completed
        )

    # =========================================================
    # BEHAVIOR-SPECIFIC PROGRESS
    # =========================================================

    def _aggressive_progress_reward(
        self,
        progress_signal,
        risk,
    ):

        profile = self.behavior_profile

        safety_gate = max(
            profile[
                "progress_safety_floor"
            ],
            1.0 - 0.20 * risk,
        )

        progress_component = (
            progress_signal
            * profile[
                "progress_distance_weight"
            ]
        )

        return (
            profile["progress_scale"]
            * progress_component
            * safety_gate
        )

    def _neutral_progress_reward(
        self,
        progress_signal,
        risk,
    ):

        profile = self.behavior_profile

        safety_gate = max(
            profile[
                "progress_safety_floor"
            ],
            1.0 - 0.40 * risk,
        )

        progress_component = (
            progress_signal
            * profile[
                "progress_distance_weight"
            ]
        )

        return (
            profile["progress_scale"]
            * progress_component
            * safety_gate
        )

    def _conservative_progress_reward(
        self,
        progress_signal,
        risk,
    ):

        profile = self.behavior_profile

        safety_gate = max(
            profile[
                "progress_safety_floor"
            ],
            1.0 - 0.60 * risk,
        )

        progress_component = (
            progress_signal
            * profile[
                "progress_distance_weight"
            ]
        )

        return (
            profile["progress_scale"]
            * progress_component
            * safety_gate
        )

    # =========================================================
    # ACTION PREFERENCE
    # =========================================================

    def _action_preference(
        self,
        action,
        risk,
    ):

        profile = self.behavior_profile

        # -----------------------------------------------------
        # Accelerate
        # -----------------------------------------------------

        if action == 3:

            score = (
                1.0
                -
                risk
                / max(
                    profile[
                        "accelerate_limit"
                    ],
                    1e-6,
                )
            )

            return float(
                np.clip(
                    score,
                    -1.0,
                    1.0,
                )
            )

        # -----------------------------------------------------
        # Maintain
        # -----------------------------------------------------

        if action == 2:

            distance_from_center = abs(
                risk
                -
                profile[
                    "maintain_center"
                ]
            )

            score = (
                1.0
                -
                distance_from_center
                /
                max(
                    profile[
                        "maintain_width"
                    ],
                    1e-6,
                )
            )

            return float(
                np.clip(
                    score,
                    -1.0,
                    1.0,
                )
            )

        # -----------------------------------------------------
        # Yield
        # -----------------------------------------------------

        if action == 1:

            start = profile[
                "yield_start"
            ]

            if risk < start:

                score = -(
                    start - risk
                ) / max(
                    start,
                    1e-6,
                )

            else:

                score = (
                    risk - start
                ) / max(
                    1.0 - start,
                    1e-6,
                )

            return float(
                np.clip(
                    score,
                    -1.0,
                    1.0,
                )
            )

        # -----------------------------------------------------
        # Brake
        # -----------------------------------------------------

        if action == 0:

            start = profile[
                "brake_start"
            ]

            if risk < start:

                score = -(
                    start - risk
                ) / max(
                    start,
                    1e-6,
                )

            else:

                score = (
                    risk - start
                ) / max(
                    1.0 - start,
                    1e-6,
                )

            # Braking becomes slightly stronger at critical risk.
            score *= 1.20

            return float(
                np.clip(
                    score,
                    -1.0,
                    1.0,
                )
            )

        raise ValueError(
            f"Invalid action: {action}"
        )

    # =========================================================
    # RISK REWARD
    # =========================================================

    def _risk_reward(
        self,
        risk,
        near_collision,
    ):

        profile = self.behavior_profile

        risk_cost = (
            risk
            **
            profile["risk_power"]
        )

        reward = -(
            profile["risk_weight"]
            * risk_cost
        )

        if near_collision:

            reward -= (
                profile[
                    "near_collision_penalty"
                ]
            )

        return float(reward)

    # =========================================================
    # RECOVERY REWARD
    # =========================================================

    def _recovery_reward(
        self,
        action,
        previous_risk,
        current_risk,
        risk_change,
    ):

        profile = self.behavior_profile

        if risk_change > profile[
            "recovery_delta"
        ]:

            return 0.0

        if current_risk > profile[
            "recovery_risk_limit"
        ]:

            return 0.0

        if action == 3:

            return profile[
                "recovery_accelerate"
            ]

        if action == 2:

            return profile[
                "recovery_maintain"
            ]

        if action == 1:

            return -profile[
                "recovery_yield"
            ]

        if action == 0:

            return -profile[
                "recovery_brake"
            ]

        return 0.0

    # =========================================================
    # LOW-SPEED / DEADLOCK
    # =========================================================

    def _low_speed_penalty(
        self,
        speed,
        risk,
    ):

        profile = self.behavior_profile

        if speed >= profile[
            "low_speed_threshold"
        ]:

            return 0.0

        # Never punish stopping when risk is actually high.
        if risk >= profile[
            "low_speed_risk_limit"
        ]:

            return 0.0

        return -profile[
            "low_speed_penalty"
        ]

    # =========================================================
    # AGGRESSIVE REWARD
    # =========================================================

    def _calculate_aggressive_reward(
        self,
        action,
        collision,
        completed,
        progress_signal,
        current_risk,
        previous_risk,
        risk_change,
        near_collision,
    ):

        profile = self.behavior_profile

        reward = 0.0

        # -----------------------------------------------------
        # Time efficiency
        # -----------------------------------------------------

        reward -= profile[
            "step_penalty"
        ]

        # -----------------------------------------------------
        # Aggressive progress
        # -----------------------------------------------------

        reward += (
            self._aggressive_progress_reward(
                progress_signal,
                current_risk,
            )
        )

        # -----------------------------------------------------
        # Risk
        # -----------------------------------------------------

        reward += (
            self._risk_reward(
                current_risk,
                near_collision,
            )
        )

        # -----------------------------------------------------
        # Action suitability
        # -----------------------------------------------------

        reward += (
            profile["action_scale"]
            *
            self._action_preference(
                action,
                current_risk,
            )
        )

        # -----------------------------------------------------
        # Recovery
        # -----------------------------------------------------

        reward += (
            self._recovery_reward(
                action,
                previous_risk,
                current_risk,
                risk_change,
            )
        )

        # -----------------------------------------------------
        # Discourage unnecessary defensive behavior
        # -----------------------------------------------------

        if current_risk < 0.30:

            if action == 1:

                reward -= profile[
                    "unnecessary_yield_penalty"
                ]

            elif action == 0:

                reward -= profile[
                    "unnecessary_brake_penalty"
                ]

        # -----------------------------------------------------
        # Deadlock protection
        # -----------------------------------------------------

        reward += (
            self._low_speed_penalty(
                self.learning_vehicle.speed,
                current_risk,
            )
        )

        # -----------------------------------------------------
        # Terminal events
        # -----------------------------------------------------

        if collision:

            reward -= profile[
                "collision_penalty"
            ]

        if completed:

            reward += profile[
                "completion_bonus"
            ]

        return float(reward)

    # =========================================================
    # NEUTRAL REWARD
    # =========================================================

    def _calculate_neutral_reward(
        self,
        action,
        collision,
        completed,
        progress_signal,
        current_risk,
        previous_risk,
        risk_change,
        near_collision,
    ):

        profile = self.behavior_profile

        reward = 0.0

        # Time efficiency.
        reward -= profile[
            "step_penalty"
        ]

        # Neutral progress.
        reward += (
            self._neutral_progress_reward(
                progress_signal,
                current_risk,
            )
        )

        # Neutral safety sensitivity.
        reward += (
            self._risk_reward(
                current_risk,
                near_collision,
            )
        )

        # Action suitability.
        reward += (
            profile["action_scale"]
            *
            self._action_preference(
                action,
                current_risk,
            )
        )

        # Recovery after risk clears.
        reward += (
            self._recovery_reward(
                action,
                previous_risk,
                current_risk,
                risk_change,
            )
        )

        # Avoid unnecessary defensive behavior.
        if current_risk < 0.30:

            if action == 1:

                reward -= profile[
                    "unnecessary_yield_penalty"
                ]

            elif action == 0:

                reward -= profile[
                    "unnecessary_brake_penalty"
                ]

        # Deadlock protection.
        reward += (
            self._low_speed_penalty(
                self.learning_vehicle.speed,
                current_risk,
            )
        )

        # Terminal events.
        if collision:

            reward -= profile[
                "collision_penalty"
            ]

        if completed:

            reward += profile[
                "completion_bonus"
            ]

        return float(reward)

    # =========================================================
    # CONSERVATIVE REWARD
    # =========================================================

    def _calculate_conservative_reward(
        self,
        action,
        collision,
        completed,
        progress_signal,
        current_risk,
        previous_risk,
        risk_change,
        near_collision,
    ):

        profile = self.behavior_profile

        reward = 0.0

        # Time efficiency.
        reward -= profile[
            "step_penalty"
        ]

        # Conservative progress.
        reward += (
            self._conservative_progress_reward(
                progress_signal,
                current_risk,
            )
        )

        # Strongest risk sensitivity.
        reward += (
            self._risk_reward(
                current_risk,
                near_collision,
            )
        )

        # Action suitability.
        reward += (
            profile["action_scale"]
            *
            self._action_preference(
                action,
                current_risk,
            )
        )

        # Strong recovery behavior prevents
        # conservative policy from stopping permanently.
        reward += (
            self._recovery_reward(
                action,
                previous_risk,
                current_risk,
                risk_change,
            )
        )

        # Defensive behavior is allowed more often,
        # but not when the state is clearly safe.
        if current_risk < 0.30:

            if action == 1:

                reward -= profile[
                    "unnecessary_yield_penalty"
                ]

            elif action == 0:

                reward -= profile[
                    "unnecessary_brake_penalty"
                ]

        # Safe-state deadlock protection.
        reward += (
            self._low_speed_penalty(
                self.learning_vehicle.speed,
                current_risk,
            )
        )

        # Terminal events.
        if collision:

            reward -= profile[
                "collision_penalty"
            ]

        if completed:

            reward += profile[
                "completion_bonus"
            ]

        return float(reward)

    # =========================================================
    # REWARD DISPATCH
    # =========================================================

    def _calculate_reward(
        self,
        action,
        collision,
        completed,
        progress_signal,
        current_risk,
        previous_risk,
        risk_change,
        near_collision,
    ):

        if self.behavior == "aggressive":

            return (
                self._calculate_aggressive_reward(
                    action=action,
                    collision=collision,
                    completed=completed,
                    progress_signal=progress_signal,
                    current_risk=current_risk,
                    previous_risk=previous_risk,
                    risk_change=risk_change,
                    near_collision=near_collision,
                )
            )

        if self.behavior == "neutral":

            return (
                self._calculate_neutral_reward(
                    action=action,
                    collision=collision,
                    completed=completed,
                    progress_signal=progress_signal,
                    current_risk=current_risk,
                    previous_risk=previous_risk,
                    risk_change=risk_change,
                    near_collision=near_collision,
                )
            )

        if self.behavior == "conservative":

            return (
                self._calculate_conservative_reward(
                    action=action,
                    collision=collision,
                    completed=completed,
                    progress_signal=progress_signal,
                    current_risk=current_risk,
                    previous_risk=previous_risk,
                    risk_change=risk_change,
                    near_collision=near_collision,
                )
            )

        raise ValueError(
            f"Unknown behavior: {self.behavior}"
        )

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_observation(self):

        v = self.learning_vehicle

        conflict = (
            self._get_conflict_risk()
        )

        conflict_risk = (
            conflict["risk"]
        )

        conflict_ttc = (
            conflict["ttc"]
        )

        if not np.isfinite(
            conflict_ttc
        ):

            conflict_ttc = 10.0

        conflict_ttc_norm = np.clip(
            conflict_ttc / 10.0,
            0.0,
            1.0,
        )

        min_distance = minimum_distance(
            v,
            self.vehicles,
        )

        min_ttc = minimum_ttc(
            v,
            self.vehicles,
        )

        if not np.isfinite(
            min_distance
        ):

            min_distance = (
                self.spawn_distance
            )

        if not np.isfinite(
            min_ttc
        ):

            min_ttc = 10.0

        min_distance_norm = np.clip(
            min_distance
            / self.spawn_distance,
            0.0,
            1.0,
        )

        min_ttc_norm = np.clip(
            min_ttc
            / 10.0,
            0.0,
            1.0,
        )

        intersection_distance = (
            math.hypot(
                v.x,
                v.y,
            )
        )

        intersection_distance_norm = np.clip(
            intersection_distance
            / self.spawn_distance,
            0.0,
            1.0,
        )

        # -----------------------------------------------------
        # Own state
        # -----------------------------------------------------

        obs = [

            v.x / self.spawn_distance,

            v.y / self.spawn_distance,

            v.speed / self.max_speed,

            v.acceleration / self.max_acceleration,

            float(v.lane),

            self._direction_value(
                v.direction
            ),

            self._route_value(
                v.route
            ),

            float(conflict_risk),

            conflict_ttc_norm,

            intersection_distance_norm,
        ]

        # -----------------------------------------------------
        # Nearby vehicles
        # -----------------------------------------------------

        others = [
            vehicle
            for vehicle in self.vehicles
            if not vehicle.is_learning
        ]

        others.sort(
            key=lambda vehicle:
            v.distance_to(vehicle)
        )

        # 5 × 6 = 30 additional values.
        for other in others[:5]:

            dx = (
                other.x - v.x
            ) / self.spawn_distance

            dy = (
                other.y - v.y
            ) / self.spawn_distance

            relative_speed = (
                other.speed - v.speed
            ) / self.max_speed

            distance_norm = np.clip(
                v.distance_to(other)
                / self.spawn_distance,
                0.0,
                1.0,
            )

            conflict_flag = float(
                self._routes_conflict(
                    v,
                    other,
                )
            )

            ego_eta, _ = (
                self._eta_to_conflict_zone(
                    v
                )
            )

            other_eta, _ = (
                self._eta_to_conflict_zone(
                    other
                )
            )

            if (
                np.isfinite(ego_eta)
                and
                np.isfinite(other_eta)
            ):

                eta_gap = abs(
                    ego_eta
                    - other_eta
                )

            else:

                eta_gap = 10.0

            eta_gap_norm = np.clip(
                eta_gap / 10.0,
                0.0,
                1.0,
            )

            obs.extend(
                [
                    dx,
                    dy,
                    relative_speed,
                    distance_norm,
                    conflict_flag,
                    eta_gap_norm,
                ]
            )

        # -----------------------------------------------------
        # Exact 40-dimensional observation
        # -----------------------------------------------------

        while len(obs) < self.observation_dim:
            obs.append(0.0)

        return np.asarray(
            obs[:self.observation_dim],
            dtype=np.float32,
        )

    # =========================================================
    # ENCODING
    # =========================================================

    @staticmethod
    def _direction_value(
        direction,
    ):

        values = {

            "north": 0.0,

            "south": 0.33,

            "east": 0.66,

            "west": 1.0,
        }

        return values[direction]

    @staticmethod
    def _route_value(
        route,
    ):

        values = {

            "straight": 0.0,

            "left": 0.5,

            "right": 1.0,
        }

        return values[route]