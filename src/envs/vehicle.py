import math


class Vehicle:

    def __init__(
        self,
        vehicle_id,
        x,
        y,
        speed,
        direction,
        lane,
        route,
        is_learning=False,
    ):
        self.vehicle_id = vehicle_id

        self.x = float(x)
        self.y = float(y)

        self.speed = float(speed)
        self.acceleration = 0.0

        self.direction = direction
        self.lane = lane
        self.route = route

        self.is_learning = is_learning

        self.collided = False
        self.completed = False

        self.length = 4.5
        self.width = 1.8

        self.in_intersection = False
        self.has_turned = False

        self.route_progress = 0.0

        self.heading = self._initial_heading()

        self.turn_radius = 10.0
        self.turn_center_x = None
        self.turn_center_y = None
        self.turn_start_angle = None
        self.turn_angle = 0.0
        self.turn_sign = 0

    # =====================================================
    # UPDATE
    # =====================================================

    def update(
        self,
        acceleration,
        dt,
    ):
        self.acceleration = float(
            acceleration
        )

        self.speed += (
            self.acceleration * dt
        )

        self.speed = max(
            0.0,
            self.speed,
        )

        distance = self.speed * dt

        # -------------------------------------------------
        # APPROACH
        # -------------------------------------------------

        if not self.in_intersection:

            self._move_forward(
                distance
            )

            if self._at_intersection():

                self.in_intersection = True

                if self.route in (
                    "left",
                    "right",
                ):
                    self._initialize_turn()

        # -------------------------------------------------
        # INTERSECTION
        # -------------------------------------------------

        elif not self.has_turned:

            if self.route == "straight":

                self._move_forward(
                    distance
                )

                if not self._inside_intersection():

                    self.has_turned = True
                    self.route_progress = 1.0

            else:

                self._update_turn(
                    distance
                )

        # -------------------------------------------------
        # AFTER INTERSECTION
        # -------------------------------------------------

        else:

            self._move_forward(
                distance
            )

        self._check_completion()

    # =====================================================
    # HEADING
    # =====================================================

    def _initial_heading(self):

        headings = {
            "north": math.pi / 2.0,
            "south": -math.pi / 2.0,
            "east": 0.0,
            "west": math.pi,
        }

        return headings[self.direction]

    # =====================================================
    # FORWARD MOTION
    # =====================================================

    def _move_forward(
        self,
        distance,
    ):

        self.x += (
            math.cos(self.heading)
            * distance
        )

        self.y += (
            math.sin(self.heading)
            * distance
        )

    # =====================================================
    # TURN INITIALIZATION
    # =====================================================

    def _initialize_turn(self):

        heading = self.heading

        forward_x = math.cos(heading)
        forward_y = math.sin(heading)

        left_x = -forward_y
        left_y = forward_x

        right_x = forward_y
        right_y = -forward_x

        if self.route == "left":

            self.turn_center_x = (
                self.x
                + left_x * self.turn_radius
            )

            self.turn_center_y = (
                self.y
                + left_y * self.turn_radius
            )

            self.turn_sign = 1

        else:

            self.turn_center_x = (
                self.x
                + right_x * self.turn_radius
            )

            self.turn_center_y = (
                self.y
                + right_y * self.turn_radius
            )

            self.turn_sign = -1

        radius_x = (
            self.x
            - self.turn_center_x
        )

        radius_y = (
            self.y
            - self.turn_center_y
        )

        self.turn_start_angle = math.atan2(
            radius_y,
            radius_x,
        )

        self.turn_angle = 0.0
        self.route_progress = 0.0

    # =====================================================
    # TURN UPDATE
    # =====================================================

    def _update_turn(
        self,
        distance,
    ):

        if self.turn_start_angle is None:
            self._initialize_turn()

        delta_angle = (
            distance
            / self.turn_radius
        )

        remaining = (
            math.pi / 2.0
            - self.turn_angle
        )

        delta_angle = min(
            delta_angle,
            remaining,
        )

        self.turn_angle += (
            delta_angle
        )

        current_angle = (
            self.turn_start_angle
            + self.turn_sign
            * self.turn_angle
        )

        self.x = (
            self.turn_center_x
            + self.turn_radius
            * math.cos(current_angle)
        )

        self.y = (
            self.turn_center_y
            + self.turn_radius
            * math.sin(current_angle)
        )

        self.heading += (
            self.turn_sign
            * delta_angle
        )

        self.route_progress = min(
            self.turn_angle
            / (math.pi / 2.0),
            1.0,
        )

        if self.turn_angle >= (
            math.pi / 2.0 - 1e-6
        ):

            self.turn_angle = (
                math.pi / 2.0
            )

            self.route_progress = 1.0
            self.has_turned = True

            self.heading %= (
                2.0 * math.pi
            )

    # =====================================================
    # INTERSECTION
    # =====================================================

    def _at_intersection(self):

        return (
            abs(self.x) <= 12.0
            and
            abs(self.y) <= 12.0
        )

    def _inside_intersection(self):

        return (
            abs(self.x) <= 14.0
            and
            abs(self.y) <= 14.0
        )

    # =====================================================
    # EXIT DIRECTION
    # =====================================================

    def _exit_direction(self):

        if self.route == "straight":
            return self.direction

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
            self.direction
        ][self.route]

    # =====================================================
    # COMPLETION
    # =====================================================

    def _check_completion(self):

        limit = 90.0

        exit_direction = (
            self._exit_direction()
        )

        if exit_direction == "north":

            if self.y >= limit:
                self.completed = True

        elif exit_direction == "south":

            if self.y <= -limit:
                self.completed = True

        elif exit_direction == "east":

            if self.x >= limit:
                self.completed = True

        elif exit_direction == "west":

            if self.x <= -limit:
                self.completed = True

    # =====================================================
    # HELPERS
    # =====================================================

    def position(self):
        return (
            self.x,
            self.y,
        )

    def distance_to(self, other):

        return math.hypot(
            self.x - other.x,
            self.y - other.y,
        )

    def speed_limit_check(
        self,
        max_speed,
    ):

        if self.speed > max_speed:
            self.speed = max_speed