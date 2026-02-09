"""
Satellite - Sputnik-like orbiting object

A small Sputnik-like satellite that moves horizontally across the cylindrical world.
Features spherical body with antennas and slow rotation.
"""

import math
import pygame
from Box2D import b2Vec2


class Satellite:
    """
    Sputnik-like satellite for orbital background element.

    Features:
    - Spherical copper/gold body (~58cm diameter like real Sputnik)
    - 4 swept-back antennas (2.4m and 2.9m like real Sputnik)
    - Constant horizontal velocity with cylindrical wrapping
    - Slow rotation for visual effect
    """

    def __init__(self, b2world, position, velocity=5.0, scale=1.0, world_width=None):
        """
        Initialize Satellite.

        Args:
            b2world: Box2D world instance
            position: (x, y) tuple in meters
            velocity: Horizontal velocity in m/s (default 5.0)
            scale: Scale factor (default 1.0)
            world_width: Width of cylindrical world for wrapping (None = no wrap)
        """
        self.scale = scale
        self.velocity = velocity
        self.world_width = world_width

        # Create Box2D body (kinematic - moves with constant velocity, no physics)
        self.body = b2world.CreateKinematicBody(
            position=position,
            angle=0.0,
        )

        # Set constant horizontal velocity
        self.body.linearVelocity = b2Vec2(velocity, 0.0)

        # Satellite dimensions (real Sputnik: 58cm diameter)
        self.radius = 0.58 / 2.0 * scale  # 0.29m radius

        # Antenna dimensions (real Sputnik had 4 antennas, 2.4m and 2.9m long)
        self.antenna_lengths = [
            2.4 * scale,
            2.9 * scale,
            2.4 * scale,
            2.9 * scale,
        ]

        # Antenna base angles (swept back from direction of travel)
        sweep_angle = math.pi * 0.3  # 54 degrees sweep back
        self.antenna_base_angles = [
            math.pi + sweep_angle * 0.6,   # Upper-right
            math.pi + sweep_angle * 1.2,   # Upper-left
            math.pi - sweep_angle * 0.6,   # Lower-right
            math.pi - sweep_angle * 1.2,   # Lower-left
        ]

        # Visual parameters
        self.rotation_angle = 0.0
        self.rotation_speed = 0.5  # radians per second

    def update(self, dt):
        """Update satellite position and handle cylindrical wrapping."""
        # Maintain constant horizontal velocity
        self.body.linearVelocity = b2Vec2(self.velocity, 0.0)

        # Update rotation for visual effect
        self.rotation_angle += self.rotation_speed * dt

        # Handle cylindrical world wrapping
        if self.world_width is not None:
            pos = self.body.position
            if pos.x > self.world_width:
                self.body.position = b2Vec2(pos.x - self.world_width, pos.y)
            elif pos.x < 0:
                self.body.position = b2Vec2(pos.x + self.world_width, pos.y)

    def draw(self, surface, cam_x, cam_y, screen_width, screen_height, ppm, world_width=None):
        """
        Draw satellite to screen with cylindrical wrapping support.

        Args:
            surface: Pygame surface
            cam_x, cam_y: Camera position in world coordinates
            screen_width, screen_height: Screen dimensions in pixels
            ppm: Pixels per meter
            world_width: World width for wrap-around drawing (optional)
        """
        # Colors - Gold/Copper like real Sputnik
        BODY_COLOR = (184, 115, 51)
        BODY_HIGHLIGHT = (255, 200, 120)
        ANTENNA_COLOR = (140, 100, 60)
        OUTLINE = (80, 50, 20)

        def w2s(world_x, world_y):
            """World to screen conversion."""
            x = screen_width / 2 + (world_x - cam_x) * ppm
            y = screen_height / 2 - (world_y - cam_y) * ppm
            return int(x), int(y)

        def draw_at_position(sat_x, sat_y):
            """Draw satellite at given world position."""
            sat_screen = w2s(sat_x, sat_y)
            radius_pixels = int(self.radius * ppm)

            # Skip if way off screen
            if sat_screen[0] < -100 or sat_screen[0] > screen_width + 100:
                return
            if sat_screen[1] < -100 or sat_screen[1] > screen_height + 100:
                return

            # Draw antennas first (behind body)
            for length, base_angle in zip(self.antenna_lengths, self.antenna_base_angles):
                total_angle = base_angle + self.rotation_angle
                dir_x = math.cos(total_angle)
                dir_y = math.sin(total_angle)

                end_x = sat_x + dir_x * length
                end_y = sat_y + dir_y * length
                end_screen = w2s(end_x, end_y)

                pygame.draw.line(surface, ANTENNA_COLOR, sat_screen, end_screen, 2)

            # Draw main spherical body
            pygame.draw.circle(surface, BODY_COLOR, sat_screen, radius_pixels, 0)
            pygame.draw.circle(surface, OUTLINE, sat_screen, radius_pixels, 1)

            # Highlight for reflective surface effect
            highlight_offset_x = int(-radius_pixels * 0.3)
            highlight_offset_y = int(-radius_pixels * 0.3)
            highlight_pos = (sat_screen[0] + highlight_offset_x, sat_screen[1] + highlight_offset_y)
            highlight_radius = max(1, int(radius_pixels * 0.4))
            pygame.draw.circle(surface, BODY_HIGHLIGHT, highlight_pos, highlight_radius, 0)

            # Panel lines on body
            left_edge = (sat_screen[0] - radius_pixels, sat_screen[1])
            right_edge = (sat_screen[0] + radius_pixels, sat_screen[1])
            pygame.draw.line(surface, OUTLINE, left_edge, right_edge, 1)

            top_edge = (sat_screen[0], sat_screen[1] - radius_pixels)
            bottom_edge = (sat_screen[0], sat_screen[1] + radius_pixels)
            pygame.draw.line(surface, OUTLINE, top_edge, bottom_edge, 1)

        # Get satellite position
        sat_pos = self.body.position

        # Draw at primary position
        draw_at_position(sat_pos.x, sat_pos.y)

        # Draw wrapped copies for cylindrical world
        ww = world_width or self.world_width
        if ww is not None:
            draw_at_position(sat_pos.x - ww, sat_pos.y)
            draw_at_position(sat_pos.x + ww, sat_pos.y)
