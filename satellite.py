"""
Satellite - WorldObject Implementation

A small Sputnik-like satellite that moves horizontally across the world.
Features spherical body with antennas.
"""

import math
import pygame
from Box2D import b2Vec2
from world import WorldObject


def local_to_world(body, local_point):
    """Convert body-local point to world coordinates."""
    return body.transform * local_point


class Satellite(WorldObject):
    """
    Sputnik-like satellite as a WorldObject.

    Features:
    - Spherical body with reflective surface
    - 4 antennas extending outward
    - Constant horizontal velocity
    - Small size (~0.6m diameter like real Sputnik)
    """

    def __init__(self, world, position, velocity=5.0, scale=1.0):
        """
        Initialize Satellite.

        Args:
            world: ApolloWorld instance
            position: (x, y) tuple or b2Vec2 position in meters
            velocity: Horizontal velocity in m/s (default 5.0)
            scale: Scale factor (default 1.0)
        """
        super().__init__(world)

        self.scale = scale
        self.velocity = velocity

        # Convert position to tuple if it's a b2Vec2
        if hasattr(position, 'x') and hasattr(position, 'y'):
            position = (position.x, position.y)

        # Create Box2D body (kinematic - moves with constant velocity)
        self.body = world.b2world.CreateKinematicBody(
            position=position,
            angle=0.0,
        )

        # Set constant horizontal velocity
        self.body.linearVelocity = b2Vec2(velocity, 0.0)

        # Satellite dimensions (real Sputnik: 58cm diameter)
        self.radius = 0.58 / 2.0 * scale  # 0.29m radius

        # Create circular fixture (for collision detection if needed)
        self.body.CreateCircleFixture(
            radius=self.radius,
            density=1.0,
            friction=0.0,
            restitution=0.0,
        )

        # Antenna dimensions (real Sputnik had 4 antennas, 2.4m and 2.9m long)
        self.antenna_lengths = [
            2.4 * scale,  # Top-right
            2.9 * scale,  # Top-left
            2.4 * scale,  # Bottom-right
            2.9 * scale,  # Bottom-left
        ]

        # Antenna base angles (swept back from direction of travel)
        # These are angles relative to backwards direction (opposite of velocity)
        sweep_angle = math.pi * 0.3  # 54 degrees sweep back
        self.antenna_base_angles = [
            math.pi + sweep_angle * 0.6,   # Upper-right, swept back
            math.pi + sweep_angle * 1.2,   # Upper-left, swept back more
            math.pi - sweep_angle * 0.6,   # Lower-right, swept back
            math.pi - sweep_angle * 1.2,   # Lower-left, swept back more
        ]

        # Visual parameters
        self.rotation_angle = 0.0  # Slowly rotate for visual effect
        self.rotation_speed = 0.5  # radians per second

    def update(self, dt):
        """Update satellite physics and position."""
        # Handle cylindrical world wrapping
        self.handle_cylindrical_wrapping(self.body)

        # Maintain constant horizontal velocity
        self.body.linearVelocity = b2Vec2(self.velocity, 0.0)

        # Update rotation for visual effect
        self.rotation_angle += self.rotation_speed * dt

    def draw(self, surface, cam_x, cam_y, screen_width, screen_height):
        """Draw satellite to screen."""
        PPM = self.world.ppm

        # Colors - Gold/Copper like real Sputnik
        BODY_COLOR = (184, 115, 51)  # Copper/gold base
        BODY_HIGHLIGHT = (255, 200, 120)  # Bright gold highlight
        ANTENNA_COLOR = (140, 100, 60)  # Darker copper for antennas
        OUTLINE = (80, 50, 20)  # Dark copper outline

        def w2s(pos):
            """World to screen conversion."""
            x = screen_width / 2 + (pos.x - cam_x) * PPM
            y = screen_height / 2 - (pos.y - cam_y) * PPM
            return int(x), int(y)

        # Get satellite world position
        sat_pos = self.body.position
        sat_screen = w2s(sat_pos)
        radius_pixels = int(self.radius * PPM)

        # Draw antennas first (behind body)
        # Antennas sweep back from direction of travel and rotate with body
        for i, (length, base_angle) in enumerate(zip(self.antenna_lengths, self.antenna_base_angles)):
            # Combine the swept-back angle with rotation
            # This keeps antennas swept back while the body rotates
            total_angle = base_angle + self.rotation_angle
            dir_x = math.cos(total_angle)
            dir_y = math.sin(total_angle)

            # Antenna endpoint
            end_x = sat_pos.x + dir_x * length
            end_y = sat_pos.y + dir_y * length
            end_screen = w2s(b2Vec2(end_x, end_y))

            # Draw antenna as thin line
            pygame.draw.line(surface, ANTENNA_COLOR, sat_screen, end_screen, 2)

        # Draw main spherical body with gradient effect
        # Outer circle (darker)
        pygame.draw.circle(surface, BODY_COLOR, sat_screen, radius_pixels, 0)
        pygame.draw.circle(surface, OUTLINE, sat_screen, radius_pixels, 1)

        # Highlight (to simulate reflective surface)
        highlight_offset_x = int(-radius_pixels * 0.3)
        highlight_offset_y = int(-radius_pixels * 0.3)
        highlight_pos = (sat_screen[0] + highlight_offset_x, sat_screen[1] + highlight_offset_y)
        highlight_radius = max(1, int(radius_pixels * 0.4))
        pygame.draw.circle(surface, BODY_HIGHLIGHT, highlight_pos, highlight_radius, 0)

        # Draw panel lines on body (simulating Sputnik's hemispherical construction)
        # Horizontal line across middle
        left_edge = (sat_screen[0] - radius_pixels, sat_screen[1])
        right_edge = (sat_screen[0] + radius_pixels, sat_screen[1])
        pygame.draw.line(surface, OUTLINE, left_edge, right_edge, 1)

        # Vertical line down middle
        top_edge = (sat_screen[0], sat_screen[1] - radius_pixels)
        bottom_edge = (sat_screen[0], sat_screen[1] + radius_pixels)
        pygame.draw.line(surface, OUTLINE, top_edge, bottom_edge, 1)
