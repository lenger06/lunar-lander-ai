"""
Apollo Terrain Generation - Atari Lunar Lander Style

Creates varied, procedural terrain with flat landing pads similar to the
classic 1979 Atari Lunar Lander arcade game.

Features:
- Jagged, mountainous terrain
- Multiple landing pads of varying difficulty
- Realistic lunar surface appearance
- Configurable difficulty levels
"""

import random
import math
from Box2D import b2Vec2


class ApolloTerrain:
    """Generate and manage terrain for Apollo lander training."""

    def __init__(self, world_width_meters=200.0, difficulty=1):
        """
        Initialize terrain generator.

        Args:
            world_width_meters: Total width of world in meters
            difficulty: 1-3, affects terrain roughness and pad size
        """
        self.world_width = world_width_meters
        self.difficulty = difficulty

    def generate_terrain(self, world, num_segments=None):
        """
        Generate Atari-style varied terrain with landing pads.

        Args:
            world: Box2D world
            num_segments: Number of terrain segments (auto-calculated if None)

        Returns:
            (terrain_body, terrain_points, pads_info)
            - terrain_body: Box2D static body
            - terrain_points: List of (x, y) tuples
            - pads_info: List of pad dicts with x1, x2, y, mult
        """
        # Auto-calculate segments based on world width for consistent detail
        if num_segments is None:
            num_segments = int(self.world_width / 2.5)  # ~2.5m per segment

        dx = self.world_width / num_segments
        heights = []

        # Initial height
        y = 6.0 + random.uniform(-2.0, 2.0)

        # Roughness increases with difficulty
        roughness = 2.5 + self.difficulty * 0.5

        # Generate varied terrain heights
        for i in range(num_segments + 1):
            # Random step
            step = random.uniform(-roughness, roughness)

            # Occasional large jumps for drama (like Atari)
            if random.random() < 0.2:
                step *= 2.5

            # Apply step and clamp
            y = max(2.0, min(15.0, y + step))
            heights.append(y)

        # Define landing pad locations
        # Format: {"start": segment_index, "len": num_segments, "mult": score_multiplier}
        base_pad_specs = self._generate_pad_specs(num_segments)

        # Ensure minimum pad width for Apollo lander (needs ~5m)
        min_pad_width_m = 5.0
        min_pad_segments = max(2, int(min_pad_width_m / dx))

        pad_specs = []
        for spec in base_pad_specs:
            new_len = max(spec["len"], min_pad_segments)
            pad_specs.append({
                "start": spec["start"],
                "len": new_len,
                "mult": spec["mult"],
            })

        # Flatten terrain for landing pads
        for spec in pad_specs:
            start = spec["start"]
            length = spec["len"]
            # Use height of first segment in pad
            pad_h = heights[start]
            for i in range(start, start + length + 1):
                if 0 <= i < len(heights):
                    heights[i] = pad_h

        # Create terrain points
        terrain_points = [(i * dx, heights[i]) for i in range(num_segments + 1)]

        # Create Box2D static body
        body = world.CreateStaticBody(position=(0, 0))

        # Create edge fixtures for each segment
        for i in range(num_segments):
            x1, y1 = terrain_points[i]
            x2, y2 = terrain_points[i + 1]

            # Check if this segment is part of a landing pad
            pad_for_segment = None
            for spec in pad_specs:
                if spec["start"] <= i < spec["start"] + spec["len"]:
                    pad_for_segment = spec
                    break

            # Create edge fixture
            fix = body.CreateEdgeFixture(
                vertices=[(x1, y1), (x2, y2)],
                density=0.0,
                friction=0.9 if pad_for_segment else 0.6,  # Pads have higher friction
            )

            # Tag fixtures with metadata
            if pad_for_segment:
                pad_start = pad_for_segment["start"]
                pad_len = pad_for_segment["len"]
                pad_x1 = terrain_points[pad_start][0]
                pad_x2 = terrain_points[pad_start + pad_len][0]
                pad_center_x = (pad_x1 + pad_x2) / 2.0
                fix.userData = {
                    "type": "pad",
                    "mult": pad_for_segment["mult"],
                    "center_x": pad_center_x
                }
            else:
                fix.userData = {"type": "terrain"}

        # Build pads_info for return
        pads_info = []
        for spec in pad_specs:
            start = spec["start"]
            length = spec["len"]
            x1 = terrain_points[start][0]
            x2 = terrain_points[start + length][0]
            y = terrain_points[start][1]
            pads_info.append({
                "x1": x1,
                "x2": x2,
                "y": y,
                "mult": spec["mult"]
            })

        return body, terrain_points, pads_info

    def _generate_pad_specs(self, num_segments):
        """
        Generate landing pad specifications based on difficulty.

        Returns list of pad specs: {"start": int, "len": int, "mult": int}
        Higher mult = harder landing, higher score
        """
        if self.difficulty == 1:
            # Easy: 3 pads, mostly large and easy to hit
            return [
                {"start": int(num_segments * 0.15), "len": 5, "mult": 2},  # Large easy pad
                {"start": int(num_segments * 0.50), "len": 4, "mult": 3},  # Medium pad
                {"start": int(num_segments * 0.80), "len": 3, "mult": 4},  # Smaller pad
            ]
        elif self.difficulty == 2:
            # Medium: 4 pads, varied sizes
            return [
                {"start": int(num_segments * 0.12), "len": 4, "mult": 2},
                {"start": int(num_segments * 0.35), "len": 3, "mult": 4},
                {"start": int(num_segments * 0.60), "len": 4, "mult": 3},
                {"start": int(num_segments * 0.85), "len": 2, "mult": 5},  # Small, hard
            ]
        else:  # difficulty >= 3
            # Hard: 5 pads, mostly small and challenging (like Atari hard mode)
            return [
                {"start": int(num_segments * 0.10), "len": 3, "mult": 3},
                {"start": int(num_segments * 0.28), "len": 2, "mult": 5},  # Small
                {"start": int(num_segments * 0.50), "len": 4, "mult": 2},  # One easier pad
                {"start": int(num_segments * 0.70), "len": 2, "mult": 6},  # Very small
                {"start": int(num_segments * 0.90), "len": 2, "mult": 5},
            ]


# -------------------------------------------------
# Terrain utility functions
# -------------------------------------------------
def get_terrain_height_at(x, terrain_points):
    """
    Get terrain height at specific x coordinate via linear interpolation.

    Args:
        x: X coordinate in meters
        terrain_points: List of (x, y) tuples

    Returns:
        Interpolated height at x
    """
    if not terrain_points or len(terrain_points) == 0:
        return 0.0

    # Clamp to terrain bounds
    if x < terrain_points[0][0]:
        return terrain_points[0][1]
    if x > terrain_points[-1][0]:
        return terrain_points[-1][1]

    # Find surrounding points and interpolate
    for i in range(len(terrain_points) - 1):
        x1, y1 = terrain_points[i]
        x2, y2 = terrain_points[i + 1]
        if x1 <= x <= x2:
            # Linear interpolation
            t = (x - x1) / (x2 - x1) if x2 != x1 else 0
            return y1 + t * (y2 - y1)

    return terrain_points[-1][1]


def calculate_terrain_clearances(lander_x, lander_y, terrain_points, offsets=None):
    """
    Calculate terrain clearances at multiple points around lander.

    This provides spatial awareness for obstacle avoidance, similar to
    bird flocking repulsion forces.

    Args:
        lander_x: Lander X position in meters
        lander_y: Lander Y position in meters
        terrain_points: List of (x, y) tuples
        offsets: List of X offsets to sample (default: [-20, -10, 0, 10, 20])

    Returns:
        List of clearance values in meters
    """
    if offsets is None:
        offsets = [-20.0, -10.0, 0.0, 10.0, 20.0]

    clearances = []
    for offset_x in offsets:
        sample_x = lander_x + offset_x
        terrain_height = get_terrain_height_at(sample_x, terrain_points)
        clearance = lander_y - terrain_height
        clearances.append(clearance)

    return clearances


def find_nearest_pad(lander_x, pads_info):
    """
    Find the nearest landing pad to lander's current position.

    Args:
        lander_x: Lander X position
        pads_info: List of pad dicts with x1, x2, y

    Returns:
        (pad_index, pad_dict, distance) or (None, None, float('inf'))
    """
    nearest_idx = None
    nearest_pad = None
    min_distance = float('inf')

    for idx, pad in enumerate(pads_info):
        pad_center_x = (pad['x1'] + pad['x2']) / 2.0
        distance = abs(lander_x - pad_center_x)
        if distance < min_distance:
            min_distance = distance
            nearest_idx = idx
            nearest_pad = pad

    return nearest_idx, nearest_pad, min_distance


def get_pad_width(pad):
    """Get width of landing pad in meters."""
    return abs(pad['x2'] - pad['x1'])


def is_point_on_pad(x, y, pad, tolerance_x=0.5, tolerance_y=0.5):
    """
    Check if point (x, y) is on landing pad.

    Args:
        x, y: Point coordinates
        pad: Pad dict with x1, x2, y
        tolerance_x: Horizontal tolerance in meters
        tolerance_y: Vertical tolerance in meters

    Returns:
        True if point is on pad
    """
    on_pad_x = (pad['x1'] - tolerance_x) <= x <= (pad['x2'] + tolerance_x)
    on_pad_y = abs(y - pad['y']) <= tolerance_y
    return on_pad_x and on_pad_y


def get_pad_difficulty_color(mult):
    """
    Get color for pad based on difficulty multiplier.

    Returns RGB tuple for rendering.
    """
    if mult >= 6:
        return (255, 100, 100)  # Red - very hard
    elif mult >= 5:
        return (255, 165, 0)    # Orange - hard
    elif mult >= 4:
        return (255, 215, 0)    # Gold - medium-hard
    elif mult >= 3:
        return (255, 255, 100)  # Yellow - medium
    else:
        return (100, 255, 100)  # Green - easy


def draw_terrain_pygame(surface, terrain_points, pads_info, cam_x, ppm, screen_width, screen_height, cam_y=0.0):
    """
    Draw terrain and landing pads using pygame.

    Args:
        surface: Pygame surface
        terrain_points: List of (x, y) tuples
        pads_info: List of pad dicts
        cam_x: Camera X position (for scrolling)
        ppm: Pixels per meter
        screen_width, screen_height: Screen dimensions
        cam_y: Camera Y position (for vertical scrolling)
    """
    import pygame

    def world_to_screen(pos):
        sx = (pos[0] - cam_x) * ppm + screen_width / 2
        sy = screen_height / 2 - (pos[1] - cam_y) * ppm  # Camera-centered Y
        return int(sx), int(sy)

    # Draw terrain lines
    if len(terrain_points) > 1:
        screen_pts = [world_to_screen(p) for p in terrain_points]
        # Filter out off-screen points for performance
        visible_pts = [pt for pt in screen_pts if -100 <= pt[0] <= screen_width + 100]
        if len(visible_pts) > 1:
            pygame.draw.lines(surface, (200, 200, 200), False, visible_pts, 2)

    # Draw landing pads with color-coding by difficulty
    for pad in pads_info:
        p1 = world_to_screen((pad['x1'], pad['y']))
        p2 = world_to_screen((pad['x2'], pad['y']))

        # Only draw if visible
        if -100 <= p1[0] <= screen_width + 100 or -100 <= p2[0] <= screen_width + 100:
            color = get_pad_difficulty_color(pad['mult'])
            pygame.draw.line(surface, color, p1, p2, 6)

            # Draw pad center marker
            pad_center = ((pad['x1'] + pad['x2']) / 2.0, pad['y'])
            pc_screen = world_to_screen(pad_center)
            pygame.draw.circle(surface, color, pc_screen, 8)

            # Draw multiplier text
            import pygame.font
            font = pygame.font.SysFont("Arial", 14, bold=True)
            mult_text = font.render(f"x{pad['mult']}", True, (255, 255, 255))
            surface.blit(mult_text, (pc_screen[0] - 10, pc_screen[1] - 25))
