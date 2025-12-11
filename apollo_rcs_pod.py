"""
Apollo RCS Pod - Reusable RCS Thruster Pod Component

A quad-cluster RCS pod with 3 nozzles (up, down, side) used on both
the Apollo Lunar Module Ascent Stage and the Command/Service Module.
"""

import pygame
from Box2D import b2Vec2


def rotate_vec(v, angle):
    """Rotate a 2D vector by angle (radians)."""
    import math
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return b2Vec2(
        v.x * cos_a - v.y * sin_a,
        v.x * sin_a + v.y * cos_a,
    )


def local_to_world(body, local_point):
    """Convert body-local point to world coordinates."""
    return body.transform * local_point


class RCSPod:
    """
    Reusable RCS Pod with 3 thrusters (up, down, side).

    Features:
    - Rectangular pod body
    - 3 triangular nozzles pointing up, down, and to the side
    - Flame rendering for active thrusters
    - Configurable position, orientation, and scale
    """

    def __init__(self, center, side_direction, scale=1.0):
        """
        Initialize RCS pod.

        Args:
            center: b2Vec2 position in body-local coordinates
            side_direction: +1 for right-pointing side thruster, -1 for left
            scale: Scale factor for sizing
        """
        self.center = center
        self.side_direction = side_direction  # +1 or -1
        self.scale = scale

        # Pod dimensions (real Apollo RCS quad: 0.67m × 0.82m)
        self.half_w = (0.67 / 2.0) * scale  # 0.335m half-width
        self.half_h = (0.82 / 2.0) * scale  # 0.41m half-height

        # Nozzle dimensions (individual thruster: 0.36m long × 0.14m diameter)
        self.nozzle_length = 0.36 * scale  # 0.36m length
        self.nozzle_base_half = (0.14 / 2.0) * scale  # 0.07m half-diameter

        # Colors
        self.pod_color = (110, 110, 110)
        self.nozzle_color = (180, 180, 180)
        self.flame_color = (255, 180, 80)

        # Flame dimensions
        self.flame_length = 0.8 * scale
        self.flame_width = (0.14 / 2.0) * scale  # Match nozzle diameter

        # Thruster definitions (name, position offset, direction)
        # PHYSICS: Thrust direction is OPPOSITE to nozzle direction (Newton's 3rd Law)
        # - Nozzle points UP → thrust pushes DOWN (dir = -Y)
        # - Nozzle points DOWN → thrust pushes UP (dir = +Y)
        # - Nozzle points SIDE → thrust pushes opposite side
        self.thrusters = {
            "UP": {
                "pos": self.center + b2Vec2(0.0, self.half_h),
                "dir": b2Vec2(0.0, -1.0),  # Nozzle up, thrust down
            },
            "DOWN": {
                "pos": self.center + b2Vec2(0.0, -self.half_h),
                "dir": b2Vec2(0.0, 1.0),  # Nozzle down, thrust up
            },
            "SIDE": {
                "pos": self.center + b2Vec2(self.side_direction * self.half_w, 0.0),
                "dir": b2Vec2(-self.side_direction, 0.0),  # Nozzle side, thrust opposite
            },
        }

    def draw_pod(self, surface, body, cam_x, cam_y, screen_width, screen_height, ppm):
        """Draw the pod body rectangle."""
        def w2s(pos):
            x = screen_width / 2 + (pos.x - cam_x) * ppm
            y = screen_height / 2 - (pos.y - cam_y) * ppm
            return int(x), int(y)

        # Pod rectangle corners
        corners_local = [
            self.center + b2Vec2(-self.half_w, -self.half_h),
            self.center + b2Vec2(self.half_w, -self.half_h),
            self.center + b2Vec2(self.half_w, self.half_h),
            self.center + b2Vec2(-self.half_w, self.half_h),
        ]

        corners_world = [local_to_world(body, v) for v in corners_local]
        pygame.draw.polygon(surface, self.pod_color, [w2s(p) for p in corners_world])

    def draw_nozzles(self, surface, body, cam_x, cam_y, screen_width, screen_height, ppm):
        """Draw the 3 thruster nozzles."""
        def w2s(pos):
            x = screen_width / 2 + (pos.x - cam_x) * ppm
            y = screen_height / 2 - (pos.y - cam_y) * ppm
            return int(x), int(y)

        # UP nozzle
        tip_up = self.center + b2Vec2(0.0, self.half_h)
        base_mid_up = self.center + b2Vec2(0.0, self.half_h + self.nozzle_length)
        up_tri_local = [
            tip_up,
            base_mid_up + b2Vec2(-self.nozzle_base_half, 0.0),
            base_mid_up + b2Vec2(self.nozzle_base_half, 0.0),
        ]
        up_tri_world = [local_to_world(body, v) for v in up_tri_local]
        pygame.draw.polygon(surface, self.nozzle_color, [w2s(p) for p in up_tri_world])

        # DOWN nozzle
        tip_down = self.center + b2Vec2(0.0, -self.half_h)
        base_mid_down = self.center + b2Vec2(0.0, -self.half_h - self.nozzle_length)
        down_tri_local = [
            tip_down,
            base_mid_down + b2Vec2(self.nozzle_base_half, 0.0),
            base_mid_down + b2Vec2(-self.nozzle_base_half, 0.0),
        ]
        down_tri_world = [local_to_world(body, v) for v in down_tri_local]
        pygame.draw.polygon(surface, self.nozzle_color, [w2s(p) for p in down_tri_world])

        # SIDE nozzle
        tip_side = self.center + b2Vec2(self.side_direction * self.half_w, 0.0)
        base_mid_side = self.center + b2Vec2(
            self.side_direction * (self.half_w + self.nozzle_length), 0.0
        )
        side_tri_local = [
            tip_side,
            base_mid_side + b2Vec2(0.0, -self.nozzle_base_half),
            base_mid_side + b2Vec2(0.0, self.nozzle_base_half),
        ]
        side_tri_world = [local_to_world(body, v) for v in side_tri_local]
        pygame.draw.polygon(surface, self.nozzle_color, [w2s(p) for p in side_tri_world])

    def draw_flames(self, surface, body, active_thrusters, cam_x, cam_y,
                    screen_width, screen_height, ppm):
        """
        Draw flames for active thrusters.

        Args:
            surface: Pygame surface
            body: Box2D body
            active_thrusters: Set of thruster names that are firing ("UP", "DOWN", "SIDE")
            cam_x, cam_y: Camera position
            screen_width, screen_height: Screen dimensions
            ppm: Pixels per meter
        """
        def w2s(pos):
            x = screen_width / 2 + (pos.x - cam_x) * ppm
            y = screen_height / 2 - (pos.y - cam_y) * ppm
            return int(x), int(y)

        for thruster_name in active_thrusters:
            if thruster_name not in self.thrusters:
                continue

            t = self.thrusters[thruster_name]

            # VISUAL: Flame goes in OPPOSITE direction from thrust (same as nozzle points)
            # Physics thrust direction is opposite to nozzle, so negate it for flame visual
            nozzle_tip_local = t["pos"]
            thrust_dir_local = t["dir"]  # Physics direction (opposite to nozzle)
            nozzle_dir_local = b2Vec2(-thrust_dir_local.x, -thrust_dir_local.y)  # Nozzle/flame direction

            # Flame base is at the EXHAUST END of the nozzle
            base_local = nozzle_tip_local + nozzle_dir_local * self.nozzle_length

            # Transform to world space
            base_world = local_to_world(body, base_local)
            dir_world = rotate_vec(nozzle_dir_local, body.angle)

            # Flame triangle
            tip = base_world + dir_world * self.flame_length
            perp = b2Vec2(-dir_world.y, dir_world.x)
            side1 = base_world + perp * self.flame_width
            side2 = base_world - perp * self.flame_width

            pygame.draw.polygon(
                surface,
                self.flame_color,
                [w2s(tip), w2s(side1), w2s(side2)]
            )

    def draw(self, surface, body, active_thrusters, cam_x, cam_y,
             screen_width, screen_height, ppm):
        """
        Draw complete pod (body + nozzles + flames).

        Args:
            surface: Pygame surface
            body: Box2D body
            active_thrusters: Set of thruster names that are firing
            cam_x, cam_y: Camera position
            screen_width, screen_height: Screen dimensions
            ppm: Pixels per meter
        """
        self.draw_pod(surface, body, cam_x, cam_y, screen_width, screen_height, ppm)
        self.draw_nozzles(surface, body, cam_x, cam_y, screen_width, screen_height, ppm)
        if active_thrusters:
            self.draw_flames(surface, body, active_thrusters, cam_x, cam_y,
                           screen_width, screen_height, ppm)
