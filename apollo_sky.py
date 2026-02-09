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

            # Spread stars from below surface to above orbit for full sky coverage
            # Range: -20m (below ground, will be occluded) to orbit_altitude * 1.5
            y = random.uniform(-20.0, self.orbit_altitude * 1.5)

            # Random brightness with more variation
            brightness = random.randint(80, 255)

            self.stars.append((x, y, brightness))

    def get_stars(self):
        """
        Get list of stars.

        Returns:
            List of (x, y, brightness) tuples
        """
        return self.stars


def draw_sky_pygame(surface, stars, star_cylinder_ratio, star_cylinder_width, world_width,
                    _terrain_points, cam_x, cam_y, screen_width, screen_height, ppm):
    """
    Draw sky with stars and parallax effect using pygame.

    Args:
        surface: Pygame surface
        stars: List of (x, y, brightness) tuples on star cylinder
        star_cylinder_ratio: Ratio of star cylinder to world cylinder
        star_cylinder_width: Width of star cylinder in meters
        world_width: Width of world (outer cylinder) in meters
        _terrain_points: Unused (terrain drawn on top naturally occludes stars)
        cam_x: Camera X position in world coordinates
        cam_y: Camera Y position in world coordinates
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        ppm: Pixels per meter
    """

    # Pre-calculate screen bounds for visibility check
    half_width = screen_width / 2
    half_height = screen_height / 2

    # Calculate parallax camera position (stars move slower)
    parallax_cam_x = (cam_x % world_width) * star_cylinder_ratio

    # Pre-calculate visible world width to skip off-screen stars early
    visible_half_width = (screen_width / ppm) / 2 + 20  # meters, with margin

    # Draw stars with cylindrical wrapping
    for star_x_cylinder, star_y, brightness in stars:
        for panel_offset in [-star_cylinder_width, 0, star_cylinder_width]:
            star_x_on_cylinder = star_x_cylinder + panel_offset

            # Calculate star's apparent position with parallax
            star_offset_from_parallax_cam = star_x_on_cylinder - parallax_cam_x
            star_world_x = cam_x + (star_offset_from_parallax_cam / star_cylinder_ratio)

            # Quick horizontal visibility check before expensive screen transform
            if abs(star_world_x - cam_x) > visible_half_width:
                continue

            # Transform to screen coordinates
            screen_x = int((star_world_x - cam_x) * ppm + half_width)
            screen_y = int(half_height - (star_y - cam_y) * ppm)

            # Only draw if on screen
            if 0 <= screen_x < screen_width and 0 <= screen_y < screen_height:
                surface.set_at((screen_x, screen_y), (brightness, brightness, brightness))
