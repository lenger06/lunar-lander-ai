"""
Apollo Sky - Star Field with Parallax Cylinder

Generates and renders background stars on a smaller inner cylinder
to create proper parallax depth effect.

Features:
- Stars on inner cylinder (configurable ratio)
- Parallax scrolling (stars move slower than terrain)
- Cylindrical wrapping
- Terrain occlusion (stars hidden below ground)
- Configurable star density and distribution
"""

import random


class ApolloSky:
    """Generate and manage sky/star field for Apollo world."""

    def __init__(self, world_width_meters, orbit_altitude, star_cylinder_ratio=0.1, num_stars=300):
        """
        Initialize sky generator.

        Args:
            world_width_meters: Total width of world in meters (outer cylinder)
            orbit_altitude: Orbital altitude in meters (for vertical distribution)
            star_cylinder_ratio: Ratio of star cylinder to world cylinder (default 0.1 = 10%)
            num_stars: Number of stars to generate
        """
        self.world_width = world_width_meters
        self.orbit_altitude = orbit_altitude
        self.star_cylinder_ratio = star_cylinder_ratio
        self.num_stars = num_stars

        # Star cylinder dimensions
        self.star_cylinder_width = self.world_width * self.star_cylinder_ratio

        # Generate stars
        self.stars = []
        self._generate_stars()

    def _generate_stars(self):
        """Generate star positions and brightness on inner cylinder."""
        self.stars = []
        for i in range(self.num_stars):
            # Distribute stars across the star cylinder width using prime number spacing
            x = (i * 73.7) % self.star_cylinder_width

            # Spread stars from surface (0) all the way to well above orbit
            y = random.uniform(0.0, self.orbit_altitude * 2.0)

            # Random brightness
            brightness = random.randint(100, 255)

            self.stars.append((x, y, brightness))

    def get_stars(self):
        """
        Get list of stars.

        Returns:
            List of (x, y, brightness) tuples
        """
        return self.stars


def draw_sky_pygame(surface, stars, star_cylinder_ratio, star_cylinder_width, world_width,
                    terrain_points, cam_x, cam_y, screen_width, screen_height, ppm):
    """
    Draw sky with stars and parallax effect using pygame.

    Args:
        surface: Pygame surface
        stars: List of (x, y, brightness) tuples on star cylinder
        star_cylinder_ratio: Ratio of star cylinder to world cylinder
        star_cylinder_width: Width of star cylinder in meters
        world_width: Width of world (outer cylinder) in meters
        terrain_points: List of (x, y) terrain tuples for occlusion
        cam_x: Camera X position in world coordinates
        cam_y: Camera Y position in world coordinates
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        ppm: Pixels per meter
    """
    import pygame
    from apollo_terrain import get_terrain_height_at

    def world_to_screen(pos):
        sx = (pos[0] - cam_x) * ppm + screen_width / 2
        sy = screen_height / 2 - (pos[1] - cam_y) * ppm
        return int(sx), int(sy)

    # Calculate parallax camera position (stars move slower)
    # Camera on outer cylinder moves normally, but we sample stars at slower rate
    parallax_cam_x = (cam_x % world_width) * star_cylinder_ratio

    # Draw stars with cylindrical wrapping (draw 3 panels for seamless wrapping)
    for star_x_cylinder, star_y, brightness in stars:
        # Draw star in 3 positions (left, center, right) on star cylinder
        for panel_offset in [-star_cylinder_width, 0, star_cylinder_width]:
            star_x_on_cylinder = star_x_cylinder + panel_offset

            # Calculate star's apparent position with parallax
            # The star's position on screen is offset from camera by the parallax ratio
            star_offset_from_parallax_cam = star_x_on_cylinder - parallax_cam_x
            star_world_x = cam_x + (star_offset_from_parallax_cam / star_cylinder_ratio)

            # Get terrain height at star's X position to check occlusion
            terrain_height = get_terrain_height_at(star_world_x % world_width, terrain_points)

            # Only draw star if it's above the terrain
            if star_y > terrain_height:
                screen_x, screen_y = world_to_screen((star_world_x, star_y))

                # Only draw if on screen
                if -50 <= screen_x <= screen_width + 50 and -50 <= screen_y <= screen_height + 50:
                    color = (brightness, brightness, brightness)
                    pygame.draw.circle(surface, color, (screen_x, screen_y), 2)
